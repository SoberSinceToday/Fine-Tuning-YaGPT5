<h1>User Simulation Neural Network</h1>

<h2>Описание проекта</h2>

<h3>User Simulation Neural Network — Fine-Tuned YandexGPT-5 Lite 8B pretrain, моделирующая манеру переписки
пользователя.</h3>

<h2>Немного <i>корректных</i> (или почти) примеров</h2>
<blockquote>Остальные есть в <code>git_utils</code>, совсем некорректные все же не грузил)</blockquote>

<h5>Без памяти(Некоторые скриншоты могут быть ранней версией модели, дообученной на неправильной разметке -> вывод не очень):</h5>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic0.png"></img>
<hr>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/incorrect_pic2.jpg"></img>
<hr>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic8.jpg"</img>
<hr>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/incorrect_pic1.jpg"></img>
<hr> 
<h5>С памятью:</h5>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic2.png"></img>
<hr>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic4.4.png"></img>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic4.1.png"></img>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic4.3.png"></img>
<img src="https://github.com/SoberSinceToday/Fine-Tuning-YaGPT5/blob/main/git_utils/pic4.2.png"></img>
<hr>

<h2>Quick Start</h2>
<h5>При желании/что-то не запускается - используйте venv</h5>

<code>git clone https://github.com/misis-programming2024-2025/misis2024f-24-17-sunik-t-v/</code>

<code>cd misis2024f-24-17-sunik-t-v/project</code>

<code>pip install -r requirements.txt</code>

<h4>Run LLM API + GUI</h4>

<code>uvicorn app:app --reload</code>

<code>python gui.py</code>

<h4>Run model training</h4>

<h5>Create .env routed to <code>src/utils</code>" with variables: API_ID, API_HASH, CHAT_ID (Telegram's APIs)</h5>

<code>python model_training.py</code>

<h2>Основные компоненты</h2>

<h3>• Парсинг и обработка данных (<code>Pyrogram</code>)</h3>
<ul>
  <li><code>src/classes/Parser.py</code> - По заданному Telegram id/username извлекает текст, дату и
отправителя сообщений
  </li>

  <li><code>src/classes/Preprocessor.py</code> - Обрабатывает данные в соответствии с определенной логикой (Бьет сообщения на response-answer пары, отслеживает изменение темы, Так же реализована логика нахождения похожих сообщений с <code>Faiss</code>, но не использована вследствие ненадобности.)</li>
  <li><code>model_training.py</code> (10-28 строка) - Применение</li>
  <li>Сохраняют по пути <code>/data/raw_dialog</code> и <code>/data/processed</code></li>
</ul>

<h3>• Fine-Tuning YandexGPT-5 Lite 8B pretrain
<h4><code>ipynbs/model_training.ipynb</code>,<code>model_training.py</code>,<code>src/classes/Model.py</code></h3></h4>
<ul>
    <li><code>QLoRA(4bit)</code> + <code>MultiGPU</code> (2xT4 Kaggle) - Обучение получилось на 2 эпохи в силу ограничения времени в 9 часов на Kaggle (Для модели маловато, но хотелось поработать с большой pretrain)</li>
    <li>Использовался <code>SFTTrainer</code>(HuggingFace) + <code>PEFT</code>(Только на Q, K, V вследствие ограниченности ресурсов), решил использовать Instruction-Output + retrieved_context (Потому что иначе реализовывать память для проекта такого уровня слишком дорого XD) разметку</li>
    <li>Выходная модель, при запуске именно локально, "забраковалась" (cpu + offload), поэтому решил использовать <code>src/classes/CustomLogitsProcessor</code> чтобы исправить частое использование некорректных/неинформативных токенов</li>
    <li>Metrics: Human Evaluation 10/10 Потому что получилось, как и задумывалось, смешно)</li>
    <li>Модель выгружена на HuggingFace</li>
</ul>

<h3>• Инференс с помощью <code>FastAPI</code></h3>
<ul>
    <li>В <code>app.py</code> реализована простая серверная часть на локальном порту</li>
</ul>

<h3>• GUI для взаимодействия с нейросетью с помощью <code>Gradio</code></h3>
<ul>
    <li>В файле <code>gui.py</code> Реализован простой GUI</li>
</ul>
<blockquote>Если вам хочется CLI можно использовать curl/Postman :)</blockquote>

<h3>• Логгер</h3>
<ul>
<li> Инициализируется в <code>src/classes/Logger.py</code></li>
<li> Логгирует и клиентскую часть (в <code>gui.py</code>) и серверную (в <code>app.py</code>)</li>
<li> Логгирует и в файл (каждый день новый файл, файлы удаляются через месяц) и в консоль</li>
</ul>

<h3>Docker</h3>
<h4><code>.dockerignore</code> и <code>Dockerfile</code>
<ul>
<li><code>docker build -t model .</code></li>
<li><code>docker run -it -p 5000:5000 model</code></li>
</ul>

<h3>Возможности масштабирования:</h3>

<h4>• Добавление ролей(friend, parent, stranger etc.) + обучение на всех диалогах с разметкой по ролям</h3>

<h4>• LangChain/LangGraph Пайплайн

<h4>• Улучшить фильтрацию токенов (а лучше просто запускать на нормальной конфигурации)


<h2> Заключение </h2>

<h4>Модель на удивление хорошо обучилась всего на 50к парах сообщений, особенно, учитывая мой стиль письма(недописанные
слова, слова-замещения которых явно нет в pretrain данных)</h4>
<h4>Всем кому я показывал, говорили мне, что это вылитый я, а к тому же еще и смешно, так что общая оценка <h>12/10</h></h4>
