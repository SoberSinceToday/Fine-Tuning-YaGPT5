from utils import formatting_prompts_func
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training
import os, torch


class Model:
    def __init__(self, **kwargs):
        """
         model_name - hf model ref
         path_to_data
         output_directory
        :param kwargs:
        """
        self.params = {"model_name": "yandex/YandexGPT-5-Lite-8B-pretrain",
                       "path_to_data": "../data/processed",
                       "output_directory": "../trained/"}
        for i in kwargs:
            if i in self.params.keys():
                self.params[i] = kwargs[i]

        # Define BitsAndBytes cfg
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Define model
        model = AutoModelForCausalLM.from_pretrained(
            self.params["model_name"],
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=bnb_config,
            use_cache=False,
            # max_memory={0:"14GiB", 1:"14GiB", "cpu":"10GiB"}
        )
        model = prepare_model_for_kbit_training(model)
        # Define tokenizer and load data
        # Обрезаем начало, чтобы сохранять в контексте диалога последние сообщения
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["model_name"], trust_remote_code=True,
                                                       padding_side="left",
                                                       add_eos_token=True, add_bos_token=True,
                                                       use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load Data
        data = load_dataset(self.params["data_path"], split="train").map(formatting_prompts_func)
        data = data.remove_columns(['instruction', 'output', 'retrieved_context'])
        # Create configs
        peft_config = LoraConfig(
            r=8,  # Чем больше R тем больше параметров
            lora_alpha=16,  # Величина матрицы
            target_modules=["q_proj", "k_proj", "v_proj"],  # Дообучаем Q, K и V механизма внимания
            lora_dropout=0.01,  # Чтоб не переобучиться
            bias="all",  # Меняем байес во всех слоях
            task_type="CAUSAL_LM"  # Обучаем генеративную модель
        )

        training_args = SFTConfig(
            label_names=["labels"],
            output_dir=self.params["output_directory"],

            per_device_train_batch_size=1,  # Размер батча для тренировки на одном устройстве (GPU/CPU)
            per_device_eval_batch_size=1,  # Размер батча для валидации на одном устройстве

            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # must be false for DDP

            gradient_accumulation_steps=1,  # Сколько раз будет накапливаться градиент перед шагом обновления
            num_train_epochs=2.0,  # Количество эпох для тренировки модели

            # Оптимизация
            learning_rate=2e-5,  # lr
            max_grad_norm=1.0,  # Ограничение на норму градиента

            # Логирование и сохранение
            logging_strategy="steps",  # Логирование через несколько шагов
            logging_steps=500,  # Логирование каждые 500 шагов
            save_strategy="steps",  # Сохранение модели после каждого шага
            save_steps=1000,  # Каждые 1000 шагов
            save_total_limit=3,  # Не более 3 последних чекпоинтов
            save_safetensors=True,  # Используем формат .safetensors

            # Использование 16-битных вычислений для ускорения
            fp16=True,  # Испоьзуем 16-битные числа
            bf16=False,  # Обучаем не на A100(

            seed=42,

            remove_unused_columns=True,  # Убираем ненужные колонки
            report_to=None,  # Отключаем логгирование
            push_to_hub=False,  # Не пушим на hf

            # Параметры для загрузчика DataLoader
            ddp_find_unused_parameters=False,  # Для распределенного обучения
            dataloader_pin_memory=False,  # Может вызывать конфликты с device_map="auto
            skip_memory_metrics=True,  # Отключаем вычисление метрик памяти, чтобы уменьшить нагрузку на систему
            disable_tqdm=False,  # Прогресс-бар
        )
        self.trainer = SFTTrainer(model=model,
                                  peft_config=peft_config,
                                  train_dataset=data,
                                  args=training_args,
                                  )

    def train(self):
        self.trainer.train()
        peft_model_path = os.path.join(self.params['output_directory'], f"lora_model")
        self.trainer.model.save_pretrained(peft_model_path)
