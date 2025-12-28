import argparse
import asyncio
import multiprocessing as mp
import os
import signal
from typing import List, Optional

from aiokafka.admin import AIOKafkaAdminClient, NewPartitions, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError
from services import llm_worker_gemma


def _run_worker(worker_id: int, consumer_group: Optional[str], health_port: Optional[int]) -> None:
    if consumer_group:
        os.environ["KAFKA_CONSUMER_GROUP"] = consumer_group
    if health_port:
        os.environ["HEALTH_PORT"] = str(health_port)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if os.name != "nt":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, loop.stop)

    try:
        loop.run_until_complete(llm_worker_gemma.main())
    finally:
        loop.close()


def main():
    parser = argparse.ArgumentParser(description="Run multiple llm-worker instances locally")
    parser.add_argument("--workers", type=int, default=2, help="Сколько воркеров поднять (по умолчанию 2)")
    parser.add_argument(
        "--consumer-group",
        type=str,
        default=os.getenv("KAFKA_CONSUMER_GROUP", "llm_worker"),
        help="Kafka consumer group для всех воркеров",
    )
    parser.add_argument(
        "--health-port-start",
        type=int,
        default=8080,
        help="Базовый порт для health-check; далее инкремент на +1 для каждого воркера",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        type=str,
        default=os.getenv("KAFKA_SERVERS", "kafka.team8-ns.svc.cluster.local:9092"),
        help="Kafka bootstrap servers (для проверки/увеличения числа партиций)",
    )
    parser.add_argument(
        "--kafka-topic",
        type=str,
        default=os.getenv("KAFKA_TOPIC", "llm.chat.request"),
        help="Kafka topic для распределения воркеров",
    )

    args = parser.parse_args()

    async def _ensure_partitions():
        admin = AIOKafkaAdminClient(bootstrap_servers=args.kafka_bootstrap)
        await admin.start()
        try:
            desired = max(args.workers, 1)
            try:
                desc = await admin.describe_topics([args.kafka_topic])
                info = desc[0]
                if info.error:
                    raise info.error
                current = len(info.partitions)
                if desired > current:
                    await admin.create_partitions({args.kafka_topic: NewPartitions(total_count=desired)})
                    print(f"Topic {args.kafka_topic}: partitions {current}->{desired}")
                else:
                    print(f"Topic {args.kafka_topic}: partitions {current} (ok for {args.workers} workers)")
            except UnknownTopicOrPartitionError:
                try:
                    await admin.create_topics([NewTopic(name=args.kafka_topic, num_partitions=desired, replication_factor=1)])
                    print(f"Topic {args.kafka_topic} created with {desired} partitions")
                except TopicAlreadyExistsError:
                    pass
        except Exception as e:
            print(f"[WARN] cannot ensure partitions for {args.kafka_topic}: {e}")
        finally:
            await admin.close()

    asyncio.run(_ensure_partitions())

    mp.set_start_method("spawn", force=True)
    procs: List[mp.Process] = []

    try:
        for idx in range(args.workers):
            port = args.health_port_start + idx if args.health_port_start else None
            p = mp.Process(target=_run_worker, args=(idx + 1, args.consumer_group, port), daemon=True)
            p.start()
            procs.append(p)
            print(f"Started worker #{idx + 1} (PID={p.pid}, group={args.consumer_group}, health_port={port})")

        print("All workers started. Ctrl+C to stop.")
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Stopping workers...")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        print("Workers stopped.")


if __name__ == "__main__":
    main()
