import asyncio
from typing import AsyncIterator, Iterator, Optional

from gemma import gm


class AsyncChatSampler(gm.text.ChatSampler):
    """
    Расширенный ChatSampler с асинхронным стримингом токенов.
    Работает с готовым prompt: str, как и оригинальный chat().
    """

    async def stream_chat(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        sampling=None,
        images=None,
        rng=None,
    ) -> AsyncIterator[str]:
        """
        Асинхронный генератор: возвращает кусочки текста по мере генерации.
        - всегда stream=True
        - вместо print' -> yield. Ы
        """
        multi_turn = self.multi_turn
        unformatted_prompt = prompt

        prompt = gm.text._template.PROMPT.format(prompt)  # да, это internal

        if not multi_turn:
            object.__setattr__(self, 'last_state', None)
            object.__setattr__(self, 'turns', [])

        out_iter: Iterator[gm.text._sampler.SamplerOutput] = self.sampler.sample(  # type: ignore
            prompt,
            images=images,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            rng=rng,
            return_state=True,
            last_state=self.last_state,
            stream=True,
        )

        text_tokens: list[str] = []
        last_state: Optional[gm.text._sampler.SamplerOutput] = None  # type: ignore

        for state in out_iter:
            last_state = state
            text_tokens.append(state.text)

            # последний токен <end_of_turn> не стримим
            if state.text == '<end_of_turn>':
                continue

            # отдаём наружу кусочек текста
            yield state.text

            # даём event loop шанс переключиться
            await asyncio.sleep(0)

        # собираем финальный текст
        if last_state is None:
            final_text = ""
        else:
            final_text = ''.join(text_tokens).removesuffix('<end_of_turn>')

        # сохраняем turns и state, как делает ChatSampler.chat
        self.turns.append(gm.text._template.UserTurn(unformatted_prompt))
        self.turns.append(gm.text._template.ModelTurn(final_text))
        object.__setattr__(self, 'last_state', last_state.state)
