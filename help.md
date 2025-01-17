# Документация к экономической модели склада

## Основные параметры склада

*   **Общая площадь (м²)**: Общая арендуемая площадь склада в квадратных метрах. Этот параметр определяет размер склада, который будет использоваться для расчетов.
*   **Стоимость аренды (руб./м²/мес.)**: Ежемесячная арендная плата за один квадратный метр складской площади. Этот параметр напрямую влияет на ежемесячные расходы и является ключевым фактором при расчете прибыльности.
*   **Доля полезной площади (%)**: Процент полезной площади от общей площади склада (за вычетом проходов, стен и т.д). Этот параметр учитывает неиспользуемое пространство, что влияет на эффективность использования склада.

## Распределение площади

*   **Режим распределения площади**:
    *   **Ручной**: Позволяет задать площадь для каждого типа хранения вручную. Это обеспечивает гибкость и точный контроль над распределением, но требует от пользователя точного ввода данных.
    *   **Автоматический**: Площади распределяются автоматически в соответствии с заданными долями. Этот режим упрощает настройку, но не дает полного контроля над распределением площадей.
*   **Простое (м²)**: Площадь, выделенная под обычное хранение вещей. Этот тип хранения предполагает долгосрочное хранение без дополнительных услуг.
*   **Займы (м²)**: Площадь, используемая для хранения вещей, принятых под залог. Этот тип хранения связан с выдачей займов и их возвратом.
*   **VIP (м²)**: Площадь, выделенная под VIP-хранение, с улучшенными условиями. Обычно этот тип хранения предполагает более высокие тарифы.
*   **Краткосрочное (м²)**: Площадь, выделенная под краткосрочное хранение, обычно с посуточной оплатой. Этот тип хранения предназначен для клиентов, которым требуется временное размещение вещей.

## Тарифы и плотности

*   **Тариф простого (руб./м²/мес.)**: Ежемесячный тариф за хранение на обычном складе в расчете на квадратный метр. Этот тариф определяет базовый доход от хранения.
*   **Полок на 1 м²**: Количество полок, установленных на одном квадратном метре складской площади. Используется для расчета общего количества вещей, которые можно хранить на складе.
*   **Тариф краткосрочного (руб./день/м²)**: Тариф за 1 м² краткосрочного хранения в день. Используется для расчета доходов от краткосрочного хранения, часто более высокий по сравнению с простым хранением.
*   **Наценка VIP (руб./м²/мес.)**: Дополнительная наценка за хранение в VIP-секции, в дополнение к обычному тарифу. Этот параметр позволяет увеличить доходность от VIP-клиентов.

## Оценка и займы

*   **Оценка вещи (%)**: Процент от оценочной стоимости вещи, которую можно выдать в качестве займа. Этот параметр определяет размер займа, который клиент может получить, и влияет на доходность от займов.
*   **Наценка реализации (%)**: Наценка, применяемая при продаже невостребованных вещей. Рассчитывается как процент от оценочной стоимости. Этот параметр влияет на дополнительный доход от реализации невостребованных вещей.
*   **Средняя оценка вещи (руб.)**: Средняя стоимость одной вещи, принимаемой на хранение или под залог. Используется для расчета общей стоимости хранимых вещей и потенциальной выручки от реализации.
*   **Ставка займов (%/день)**: Дневная процентная ставка, начисляемая по выданным займам. Этот параметр определяет доходность от займовых операций.
*   **Средний срок займа (дней)**: Средний срок, на который обычно выдаются займы под залог вещей. Используется для расчета общего дохода от процентов по займам.

## Реализация (%)

*   **Реализация простое (%)**: Процент вещей из простого хранения, которые выставляются на реализацию (продажу) в случае невостребованности.
*   **Реализация займы (%)**: Процент вещей, оставленных в качестве залога и не выкупленных, которые выставляются на реализацию.
*   **Реализация VIP (%)**: Процент вещей из VIP-хранения, которые выставляются на реализацию.
*   **Реализация краткосрочное (%)**: Процент вещей из краткосрочного хранения, которые выставляются на реализацию.

## Процент заполняемости

*   **Заполнение простое (%)**: Процент фактического заполнения площади простого хранения. Влияет на количество хранимых вещей и доходы.
*   **Заполнение займы (%)**: Процент фактического заполнения площади, выделенной под займы.
*   **Заполнение VIP (%)**: Процент фактического заполнения VIP-секции.
*   **Заполнение краткосрочное (%)**: Процент фактического заполнения краткосрочного хранения.

## Плотность (вещей/м²)

*   **Плотность простое (вещей/м²)**: Среднее количество вещей, размещаемых на одном квадратном метре в зоне простого хранения.
*   **Плотность займы (вещей/м²)**: Среднее количество вещей, размещаемых на одном квадратном метре в зоне займов.
*   **Плотность VIP (вещей/м²)**: Среднее количество вещей, размещаемых на одном квадратном метре в VIP-секции.
*   **Плотность краткосрочное (вещей/м²)**: Среднее количество вещей, размещаемых на одном квадратном метре в зоне краткосрочного хранения.

## Отток клиентов/вещей (%)

*   **Отток простое (%)**: Ежемесячный процент клиентов, которые прекращают пользоваться услугами простого хранения.
*   **Отток займы (%)**: Ежемесячный процент оттока вещей, выданных под залог и не выкупленных.
*   **Отток VIP (%)**: Ежемесячный процент оттока клиентов из VIP-секции.
*   **Отток краткосрочное (%)**: Ежемесячный процент оттока клиентов из краткосрочного хранения.

## Финансы (ежемесячные)

*   **Зарплата (руб./мес.)**: Общая сумма расходов на зарплату персонала в месяц.
*   **Прочие (руб./мес.)**: Сумма прочих ежемесячных расходов, не включенных в другие категории, таких как расходные материалы, канцелярские расходы и т.п..
*   **Амортизация (руб./мес.)**: Ежемесячная сумма амортизации оборудования и других активов. Учитывает износ активов.
*   **Маркетинг (руб./мес.)**: Сумма расходов на маркетинг и рекламу в месяц.
*   **Страхование (руб./мес.)**: Сумма ежемесячных расходов на страхование имущества и ответственности.
*   **Налоги (руб./мес.)**: Сумма ежемесячных налоговых отчислений.
*   **Коммуналка (руб./мес.)**: Сумма ежемесячных расходов на коммунальные услуги.
*   **Обслуживание (руб./мес.)**: Сумма ежемесячных расходов на обслуживание и ремонт склада.

## Финансы (единовременные)

*   **Настройка (руб.)**: Единовременные расходы на настройку и подготовку склада к работе, например, настройка IT-систем.
*   **Оборудование (руб.)**: Единовременные расходы на приобретение оборудования для склада, такого как стеллажи, погрузчики и т.д.
*   **Другие (руб.)**: Сумма прочих единовременных расходов, не включенных в другие категории, например, расходы на получение лицензий.
*   **Юридические (руб.)**: Единовременные юридические расходы, связанные с открытием и ведением бизнеса.
*   **Логистика (руб.)**: Единовременные расходы на логистику, включая доставку и установку оборудования.

## Переменные расходы

*   **Упаковка (руб./м²)**: Стоимость упаковки товаров на 1 м² площади. Зависит от количества хранимых вещей и затрат на упаковочные материалы.
*   **Электричество (руб./м²)**: Стоимость электроэнергии, потребляемой на 1 м² площади. Зависит от энергопотребления склада.

## Инфляция и рост

*   **Инфляция (%/мес.)**: Ожидаемый ежемесячный темп инфляции, влияющий на стоимость товаров и услуг.
*   **Рост аренды (%/мес.)**: Ожидаемый ежемесячный процентный рост стоимости аренды складских помещений.
*   **Рост зарплаты (%/мес.)**: Ожидаемый ежемесячный процентный рост зарплат сотрудников.
*   **Рост прочих расходов (%/мес.)**: Ожидаемый ежемесячный процентный рост прочих расходов, влияющий на себестоимость услуг.

## Расширенные параметры

*   **Отключить расширенные параметры**: Если включено, то не используются расширенные настройки, которые влияют на прогнозирование.
*   **Горизонт прогноза (мес.)**: Количество месяцев, на которое строится прогноз. Определяет период, на который строится экономическая модель.
*   **Вероятность невозврата (%)**: Вероятность того, что займ не будет возвращен. Влияет на расчеты дохода от займовых операций.
*   **Коэффициент ликвидности**: Коэффициент, учитывающий ликвидность активов при расчете BEP. Используется для корректировки BEP. Показывает, какая часть активов компании может быть быстро превращена в денежные средства. Более высокий коэффициент ликвидности означает более быстрый выход в точку безубыточности.
*   **Коэффициент запаса**: Коэффициент, используемый для расчета необходимого запаса при расчете BEP. Увеличивает расходы для обеспечения безопасности. Более высокий коэффициент запаса увеличивает расходы, но делает более устойчивой работу бизнеса.
*   **Льготный период (мес.)**: Период, в течение которого по займам не начисляются проценты. Влияет на расчет доходов от займов в начальный период.
*   **Рост доходов (%/мес.)**: Ожидаемый ежемесячный рост доходов от всех видов хранения и займов.
*   **Рост расходов (%/мес.)**: Ожидаемый ежемесячный рост общих расходов, включая как постоянные, так и переменные расходы.
*   **Метод прогнозирования**: Метод, используемый для построения прогноза доходов и расходов.
    *   **Базовый**: Линейный прогноз на основе начальных значений доходов и расходов. Этот метод прост в использовании, но не учитывает динамику данных. Он предполагает, что доходы и расходы будут меняться линейно в течение прогнозируемого периода.
    *   **ML (линейная регрессия)**: Использует модель линейной регрессии для прогнозирования. Этот метод учитывает корреляцию между признаками (месяцем) и доходами. Он подходит для случаев, когда между переменными есть линейная зависимость и не наблюдается сложных нелинейных закономерностей.
    *   **ML (полиномиальная регрессия)**: Модель полиномиальной регрессии, которая позволяет уловить более сложные, нелинейные зависимости между признаками и целевой переменной. Степень полинома настраивается пользователем. Подходит, когда есть криволинейная зависимость между месяцем и доходами, когда линейной модели недостаточно.
    *   **Симуляция Монте-Карло**: Метод, основанный на моделировании множества сценариев для учета неопределенности. Он позволяет учесть случайные колебания в доходах и расходах, и оценить диапазон возможных результатов. Этот метод использует генератор случайных чисел для создания различных сценариев будущих доходов и расходов.
         * **Равномерное распределение**: Использует равномерное распределение случайных значений в заданном диапазоне отклонений. Все значения в диапазоне имеют равную вероятность. Этот тип распределения подходит для ситуаций, где все значения отклонений имеют одинаковую вероятность.
         * **Нормальное распределение**: Использует нормальное (гауссово) распределение случайных значений, заданных средним и стандартным отклонением. Подходит для моделирования процессов, которые тяготеют к среднему значению. Этот тип распределения наиболее распространен в природе и подходит для моделирования многих процессов.
         * **Треугольное распределение**:  Использует треугольное распределение случайных значений, заданное минимальным, наиболее вероятным (модой) и максимальным значениями. Этот метод подходит для моделирования ситуаций, где есть более вероятные значения, и менее вероятные, которые находятся ближе к границам диапазона.
    *   **ML (случайный лес)**: Метод машинного обучения, основанный на ансамбле деревьев решений. Случайный лес подходит для сложных зависимостей, когда есть много факторов и сложная динамика изменения доходов. Метод устойчив к выбросам и переобучению. Он использует много "деревьев решений", которые работают вместе и выдают результат усреднения. Метод эффективен, когда нет явных закономерностей.
    *   **ML (SVR)**: Метод опорных векторов для регрессии (Support Vector Regression). Этот метод подходит для сложных зависимостей и используется для прогнозирования как линейных, так и нелинейных зависимостей. SVR хорошо работает в случаях, где есть зашумленные данные.

*   **Симуляций Монте-Карло**: Число симуляций при использовании метода Монте-Карло. Чем больше симуляций, тем точнее результат, но дольше время расчета. Рекомендуется устанавливать значение в пределах от 100 до 1000, в зависимости от необходимой точности и вычислительных ресурсов.
*   **Отклонения (0.1 = ±10%)**: Диапазон случайных изменений при моделировании методом Монте-Карло (0.1 означает +-10%). Этот параметр определяет, насколько сильно могут меняться значения доходов и расходов в каждой симуляции, при равномерном распределении.
*   **Seed**: Зерно для генератора случайных чисел при моделировании методом Монте-Карло, для воспроизводимости результатов. Задавая определенное число, вы будете получать одни и те же случайные значения, что позволяет воспроизвести результаты моделирования.
*   **Распределение**: Тип распределения случайных величин в симуляции Монте-Карло. Определяет, как именно будет моделироваться неопределенность в данных.
    *   **Равномерное**: Случайные значения распределены равномерно в заданном диапазоне отклонений. Все значения в диапазоне имеют равную вероятность.
    *   **Нормальное**: Случайные значения распределены по нормальному (гауссову) закону, с заданными средним и стандартным отклонением. Подходит для моделирования процессов, которые тяготеют к среднему значению.
    *   **Треугольное**: Случайные значения распределены по треугольному закону, с заданными минимальным, наиболее вероятным (модой) и максимальным значениями. Этот метод подходит для моделирования ситуаций, где есть более вероятные значения, и менее вероятные, которые находятся ближе к границам диапазона.
    *   **Среднее (норм. распр.)**: Среднее значение для нормального распределения. Задает центральную точку для нормального распределения.
    *   **Ст. отклонение (норм. распр.)**: Стандартное отклонение для нормального распределения. Определяет ширину диапазона случайных величин.
    *   **Мин. значение (треуг. распр.)**: Минимальное значение для треугольного распределения.
    *   **Мода (треуг. распр.)**: Мода (наиболее вероятное значение) для треугольного распределения.
    *   **Макс. значение (треуг. распр.)**: Максимальное значение для треугольного распределения.
*   **Включить расширенный ML-прогноз**: Включает использование моделей машинного обучения для прогнозирования. Если отключено, используется простой линейный прогноз.
*  **Степень полинома**: Степень полинома при использовании полиномиальной регрессии. Указывает, насколько сложной может быть кривая, аппроксимирующая данные. Значения 1 (линейная регрессия), 2 (парабола), 3 и выше могут моделировать более сложные зависимости.
*   **Количество деревьев**: Количество деревьев, используемых в модели случайного леса. Большее количество деревьев делает модель более устойчивой, но может потребовать больше времени на обучение. Рекомендуется устанавливать значения от 100 до 500.
*   **Признаки**: Список признаков, которые используются при обучении моделей машинного обучения. Вы можете выбрать, какие из автоматический сгенерированных признаков наиболее подходят для вашей модели, или использовать все признаки.
    *   **Месяц**: Порядковый номер месяца в данных.
    *   **Lag_1**: Значение целевой переменной (дохода) в предыдущем месяце.
    *   **Lag_2**: Значение целевой переменной (дохода) за 2 месяца до текущего.
    *   **Rolling_Mean_3**: Среднее значение целевой переменной за последние 3 месяца.
    *   **Rolling_Mean_5**: Среднее значение целевой переменной за последние 5 месяцев.
*   **Поиск параметров**: Метод поиска параметров ML моделей.
    *   **Нет**: Параметры не ищутся, используются значения по умолчанию.
    *   **GridSearchCV**: Перебор всех значений из сетки. Позволяет найти оптимальные параметры, но может занимать много времени.
    *   **RandomizedSearchCV**: Случайный перебор значений из заданных распределений. Позволяет найти хорошие параметры за меньшее время.

## Основные термины:

*   **Прибыль**: Разница между доходами и расходами. **Формула**: `Прибыль = Доходы - Расходы`.
*   **Маржа прибыли**: Отношение прибыли к доходам в процентах. **Формула**: `Маржа прибыли = (Прибыль / Доходы) * 100`.
*   **Рентабельность**: Отношение прибыли к расходам в процентах. **Формула**: `Рентабельность = (Прибыль / Расходы) * 100`.
*   **ROI (Return on Investment)**: Показывает возврат на инвестиции, вычисляется как отношение разницы между доходами и расходами к расходам, умноженное на 100. **Формула**: `ROI = ((Доходы - Расходы) / Расходы) * 100`.
*   **IRR (Internal Rate of Return)**: Внутренняя норма доходности, показывает доходность проекта с учётом дисконтирования денежных потоков. IRR используется для определения того, является ли инвестиция выгодной, приравнивая текущую стоимость будущих денежных потоков к начальным инвестициям.
*   **NPV (Net Present Value)**: Чистая приведенная стоимость, показывает разницу между текущей стоимостью будущих денежных потоков и текущими инвестициями. **Формула**: `NPV = ∑ (CFt / (1+r)^t ) - C0`, где `CFt` - это денежный поток в момент времени t, `r` - ставка дисконтирования, `t` - время, `C0` - начальные инвестиции. NPV учитывает стоимость денег во времени. Положительное значение NPV означает, что проект прибыльный.
*   **BEP (Точка безубыточности)**: Минимальный уровень дохода, при котором склад не убыточен, где общие доходы равны общим расходам.

## Интерпретация результатов

### Общие результаты

* **Общий доход**: Общая сумма всех доходов, полученных от склада.
* **Общие расходы**: Общая сумма всех расходов, понесенных складом.
* **Прибыль**: Разница между общим доходом и общими расходами. Показывает, является ли бизнес прибыльным.
* **Маржа прибыли (%)**: Показывает долю прибыли в каждом рубле дохода.
* **Рентабельность (%)**: Показывает, насколько эффективно используются расходы для получения прибыли.
*   **ROI (%)**: Показывает рентабельность инвестиций в бизнес.
*   **IRR (%)**: Внутренняя норма доходности, показывает доходность проекта с учётом дисконтирования денежных потоков.
*   **NPV (руб.)**: Чистая приведенная стоимость, показывает общую текущую стоимость будущих денежных потоков, с учетом дисконтирования и начальных инвестиций.
*  **Мин. сумма займа (руб.)**: Минимальная сумма займа на одну вещь для покрытия расходов (BEP).

### Прогнозирование

*   **Базовый прогноз**: Представляет линейный прогноз доходов и расходов, основанный на текущих значениях. Это простой метод, который не учитывает динамику данных.
*   **ML-прогноз**: Прогноз, построенный с использованием моделей машинного обучения.
    *   **RMSE, R², MAE**: Метрики качества модели, которые показывают точность прогнозирования. RMSE — корень среднеквадратической ошибки, R² — коэффициент детерминации (оценка качества модели), MAE — средняя абсолютная ошибка.
    *   Если модель ML не обучена, вы увидите сообщение об этом, и нужно будет нажать кнопку "Обучить модель".
*   **Симуляция Монте-Карло**: Результаты моделирования методом Монте-Карло показывают диапазоны возможных значений доходов, расходов и прибыли, а не точные числа. Диапазоны показывают, как сильно могут варьироваться доходы и расходы в зависимости от неопределенности.
    *  Линии на графике показывают результаты отдельных симуляций, а средние значения показывают наиболее вероятный результат.
    * Чем больше симуляций, тем более точный результат.

### Точка безубыточности (BEP)

*   **Доход для BEP**: Это минимальная сумма дохода, которую должен получить склад, чтобы покрыть все расходы.
*   **Мин. залоговая сумма для BEP**: Минимальная сумма займа на одну вещь для покрытия расходов (BEP).
*   **Помесячная BEP**: Показывает, как меняется точка безубыточности в течение времени.

### Детализация

*   **Площадь (м²)**: Показывает распределение площади между различными видами хранения.
*   **Доход (руб.)**: Показывает, как формируется доход по разным видам хранения и реализации.
*   **Прибыль (руб.)**: Показывает прибыль по видам хранения и реализации.
*   **Ежемесячные расходы (руб.)**: Показывает ежемесячные расходы, связанные с разными типами хранения.

### Сценарный анализ
* Позволяет анализировать как изменится прибыль и расходы при различных сценариях (Оптимистичный, Реалистичный, Пессимистичный).
* Вы можете настраивать изменение тарифов, ставок и вероятности невозврата в каждом из сценариев и увидеть как это повлияет на итоговый результат.

### Диагностика
*  Помогает понять, как формируются результаты, что бы можно было найти и исправить ошибки при вводе параметров.

## Ограничения модели

*   **Линейность базового прогноза**: Базовый прогноз предполагает линейное изменение доходов и расходов, что может не соответствовать реальной ситуации.
*   **Зависимость ML-моделей от данных**: ML-модели работают лучше при наличии большого объема исторических данных. Качество прогноза зависит от качества и количества загруженных данных.
*   **Ограниченность модели Монте-Карло**: Монте-Карло моделирует случайные колебания, но не учитывает все возможные факторы риска (например, форс-мажорные обстоятельства).
*   **Предположения об однородности**: Модель предполагает, что все вещи для займа имеют одинаковую среднюю оценку и сроки хранения.
*   **Отсутствие учета сезонности**: Модель не учитывает сезонные колебания спроса на хранение.
*   **Отсутствие учета динамических изменений цен**: Модель не учитывает динамические изменения цен на рынке (например, цен на электричество).
*   **Ограниченность используемых признаков**: В моделях машинного обучения используется ограниченный набор фич.
*  **Сложность интерпретации ML**: Результаты ML-моделей могут быть не всегда легко интерпретируемы.

## Рекомендации по использованию

1.  **Начните с базовых параметров**: Начните с ввода основных параметров склада (площадь, аренда, тарифы).
2.  **Используйте разные методы прогнозирования**: Сравните результаты, полученные с помощью разных методов прогнозирования (базовый, ML, Монте-Карло), чтобы оценить их эффективность.
3.  **Проанализируйте точку безубыточности**: Убедитесь, что доходы покрывают расходы, и проанализируйте минимальную сумму займа для достижения BEP.
4.  **Изучите детализацию**: Используйте вкладку "Детализация", чтобы понять, какие виды хранения приносят наибольшую прибыль, и где возникают основные расходы.
5. **Экспериментируйте со сценариями**: Проведите анализ "Что, если" с помощью инструмента сценарного анализа, меняя процентные ставки и другие параметры для оптимизации.
6.  **Используйте вкладку "Диагностика"**: Проверьте все параметры и их влияние на прибыль, чтобы убедиться, что нет ошибок при вводе данных.
7.  **Анализируйте результаты**: Сравните результаты, полученные разными методами, и принимайте решения на основе наиболее достоверных данных.
8.  **Загружайте данные для ML-моделей**: Для более точного прогноза загружайте исторические данные по доходам и расходам.
9.  **Протестируйте различные типы распределения в методе Монте-Карло**: Попробуйте равномерное, нормальное и треугольное распределение для понимания влияния неопределенности на ваш прогноз.
10. **Используйте автоматический выбор параметров**: Поэкспериментируйте с разными методами поиска параметров для выбора оптимальной ML модели.
11. **Сохраняйте параметры**: Сохраняйте наиболее удачные настройки параметров для дальнейшего использования или сравнения.

## Основные термины:

*   **Прибыль**: Разница между доходами и расходами. **Формула**: `Прибыль = Доходы - Расходы`.
*   **Маржа прибыли**: Отношение прибыли к доходам в процентах. **Формула**: `Маржа прибыли = (Прибыль / Доходы) * 100`.
*   **Рентабельность**: Отношение прибыли к расходам в процентах. **Формула**: `Рентабельность = (Прибыль / Расходы) * 100`.
*   **ROI (Return on Investment)**: Показывает возврат на инвестиции, вычисляется как отношение разницы между доходами и расходами к расходам, умноженное на 100. **Формула**: `ROI = ((Доходы - Расходы) / Расходы) * 100`.
*   **IRR (Internal Rate of Return)**: Внутренняя норма доходности, показывает доходность проекта с учётом дисконтирования денежных потоков. IRR используется для определения того, является ли инвестиция выгодной, приравнивая текущую стоимость будущих денежных потоков к начальным инвестициям.
*   **NPV (Net Present Value)**: Чистая приведенная стоимость, показывает разницу между текущей стоимостью будущих денежных потоков и текущими инвестициями. **Формула**: `NPV = ∑ (CFt / (1+r)^t ) - C0`, где `CFt` - это денежный поток в момент времени t, `r` - ставка дисконтирования, `t` - время, `C0` - начальные инвестиции. NPV учитывает стоимость денег во времени. Положительное значение NPV означает, что проект прибыльный.
*  **BEP (Точка безубыточности)**: Минимальный уровень дохода, при котором склад не убыточен, где общие доходы равны общим расходам. **Формула (без амортизации единовременных расходов)**: `BEP = Общие расходы`. **Формула (с амортизацией единовременных расходов)**: `BEP = Общие расходы + (Единовременные расходы / Горизонт прогноза)`.