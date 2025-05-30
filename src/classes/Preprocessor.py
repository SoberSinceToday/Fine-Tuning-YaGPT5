import pandas as pd, numpy as np, json, faiss
from utils import *


class Preprocessor:
    def __init__(self, data: pd.DataFrame, model):
        self.outputs = []
        self.data = data
        self.processed_data = []
        self.index = None
        self.embedding_model = model
        self.instructions = []
        self.instr_out = []

    def preprocess_data(self, chat_id: int | str, sim_instr=True) -> None:
        """
        Итеративно создает instruction-output словарь
        :param sim_instr: Нужно ли находить похожие запросы и встаривать ответы на них в RAG
        :param chat_id:
        :return: None
        """
        self.data["time"] = pd.to_datetime(self.data["time"])

        instruction, output, retrieved_context = "", "", []
        prev_time = None

        for i in self.data.itertuples():
            if not prev_time:  # Если нет предыдущего сообщения
                prev_time = i.time
            else:  # Если уже диалог есть
                # Проверяем на актуальность
                if (i.time - prev_time) > pd.Timedelta(hours=3):
                    instruction, output, retrieved_context = "", "", []

            # Если инструкция уже есть
            if str(i.user) == chat_id and output:
                self.processed_data.append({'instruction': instruction.strip(), 'output': output.strip(),
                                            'retrieved_context': retrieved_context[:]})
                retrieved_context.append({'instruction': instruction.strip(), 'output': output.strip()})
                instruction, output = "", ""

            if str(i.user) == chat_id and not output:
                instruction += "\n" + i.message
            elif instruction:
                output += "\n" + i.message
            prev_time = i.time

        self.to_jsonl()
        if sim_instr:
            self.create_faiss()
            for i in range(len(self.processed_data)):
                instr = create_instr_output(self.processed_data[i]['instruction'], self.processed_data[i]['output'])
                sim = self.find_similar_instructions(instr)
                if len(sim):
                    self.processed_data[i]['similar'] = sim[1:]
                    if self.processed_data[i]['similar']: print(f"{self.processed_data[i]['instruction']} AAA {self.processed_data[i]['similar']}")
        self.to_jsonl()

    def to_jsonl(self) -> None:
        """
        Преобразует созданный словарь в .jsonl
        :return: None
        """
        with open("../project/data/processed/processed_data.jsonl", "w", encoding="utf-8") as f:
            for i in self.processed_data:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")

    def create_faiss(self):
        """
        Добавляет колонку retrieved context
        :return:
        """

        self.data = []
        self.instructions = []
        self.outputs = []
        with open("../project/data/processed/processed_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                self.instructions.append(self.data[-1]["instruction"])
                self.outputs.append(self.data[-1]["output"])

        # Объединяем инструкции с ответами и создаем векторы
        self.instr_out = [create_instr_output(x[0], x[1]) for x in zip(self.instructions, self.outputs)]
        vectors = self.embedding_model.encode(self.instr_out, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(vectors)

        # Создаём FAISS-индекс для поиска по косинусному сходству
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)  # Добавляем векторы

    def find_similar_instructions(self, instruction: str, top_k=5, threshold=0.95):
        """Находит `top_k` ближайших инструкций, но фильтрует по порогу схожести"""
        query_vector = np.array([self.embedding_model.encode(instruction)]).astype('float32')
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)
        similar = []
        for i in zip(distances[0], indices[0]):
            if i[1] != -1:
                if i[0] < threshold:
                    break
                else:
                    for j in self.data:
                        if create_instr_output(j['instruction'], j['output']) == self.instr_out[i[1]]:
                            instr, out = self.instr_out[i[1]].split("[SEP]")
                            similar.append({"instruction": instr, "output": out})

        return similar
