from src.classes.Parser import Parser
import pandas as pd


async def parse_data(parser: Parser, chat_id: str) -> None:
    """
    Считывает диалог в csv формат и сохраняет в data/raw_dialog
    :param parser: Parser-Объект
    :param chat_id: ID чата для считывания (без @)
    """
    data = await parser.parse_data(chat_id=chat_id)
    df = pd.DataFrame(data[::-1])
    df.rename(columns={"date": "time", "user": "user", "data": "message"}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('data/raw_dialog/data.csv', index=False)


def create_instr_output(instr: str, output: str) -> str:
    """Добавляет разделительный токен"""
    return instr + "[SEP]" + output


def formatting_prompts_func(data: dict, last_mes_amount=5):
    """
    Функция подготовки промпта
    data: Словарь с полями [instruction, output, retrieved_context]
    Поле retrieved_context - массив пар instruction-output
    """
    retrieved = "\n".join(
        [f"Пользователь1: {x['instruction']}\nПользователь2: {x['output']}" for x in
         data['retrieved_context'][-last_mes_amount:]])
    if retrieved:
        prompt = f"{retrieved}\nПользователь1: {data['instruction']}\nПользователь2: "
    else:
        prompt = f"Пользователь1: {data['instruction']}\nПользователь2: "
    return {'prompt': prompt, 'completion': data['output']}


def create_input(text, tokenizer, model, logits_processor):
    """Функция генерации модели"""
    input_text = text
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,  # Входные данные
        max_new_tokens=100,  # Максимальная длина генерируемого текста
        num_return_sequences=1,  # Количество возвращаемых последовательностей
        no_repeat_ngram_size=3,  # Избегаем повторений n-грамм
        temperature=0.4,  # Температура генерации (для контроля случайности)
        top_k=None,  # Ограничение на количество рассматриваемых наиболее вероятных токенов
        top_p=0.8,  # Использование вероятностного порога для выборки
        do_sample=True,  # Включить выборку (для генерации более разнообразных текстов)
        repetition_penalty=1,  # Наказание за повторы

        logits_processor=logits_processor  # logits processor
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def create_prompt(data, last_mes_amount=10):
    retrieved = "\n".join(
        [f"Пользователь1: {x['instruction']}\nПользователь2: {x['output']}" for x in
         data['retrieved_context'][-last_mes_amount:]])
    if retrieved:
        prompt = f"{retrieved}\nПользователь1: {data['instruction']}\nПользователь2: "
    else:
        prompt = f"Пользователь1: {data['instruction']}\nПользователь2: "
    return {'text': prompt}
