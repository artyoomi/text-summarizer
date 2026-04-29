// Page setup
#set page(
  width: 210mm,
  height: 297mm,
  margin: (top: 20mm, bottom: 20mm, left: 30mm, right: 15mm)
)

// General text setup
#set text(
  size: 14pt,
  lang: "ru"
)

// Paragraph setup
#set par(
  leading: 1.5em,
  first-line-indent: 1.25cm,
  justify: true
)

// To provide numeration like 1, 1.1, 1.1.1 and so on
#set enum(full: true)

// Setup level 1 header
#show heading.where(level: 1): it => [
  #set text(size: 14pt, weight: "bold")
  #set par(first-line-indent: 0pt, leading: 1.5em)
  #set align(center)
  #upper(it.body)
]

// Setup level 2 header
#show heading.where(level: 2): it => [
  #set text(size: 14pt, weight: "bold")
  #set par(first-line-indent: 1.25cm, leading: 1.5em, justify: true)
  #it.body
]

// Setup table captions
#show figure.where(kind: table): fig => {
  align(left)[
    #fig.caption
    #fig.body
  ]
}

// Long "-" between numering and caption in all figures
#show figure: set figure.caption(separator: [ ---])

// Force all raw blocks to have 1em indent between lines
#show raw.where(block: true): set par(leading: 1em)
// Force all raw blocks to have left alignment
#show raw.where(block: true): set align(left)

// Enable formula numbering
#set math.equation(numbering: "(1)")

#align(center)[
  #set text(weight: "semibold")

  #set par(leading: 1em)

  МИНОБРНАУКИ РОССИИ \
  САНКТ-ПЕТЕРБУРГСКИЙ ГОСУДАРСТВЕННЫЙ \
  ЭЛЕКТРОТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ \
  «ЛЭТИ» ИМ. В.И. УЛЬЯНОВА (ЛЕНИНА) \
  Кафедра МО ЭВМ

  #v(54mm)

  ОТЧЕТ \
  по лабораторной работе №3 \
  по дисциплине «Интеллектуальные технологии и компьютерные инструменты передачи и извлечения знаний» \
  Тема: Автоматическое реферирование текста

  #v(54mm)

  #table(
    columns: (33%, 33%, 33%),
    inset: 10pt,
    align: horizon,
    stroke: none,
    "Студент гр. 3381",
    "",
    table.hline(start: 1 , end: 2),
    "Иванов А.А.",
    "Преподаватель",
    "",
    table.hline(start: 1 , end: 2),
    "Малютин Е.В."
  )

  #set align(bottom)
  Санкт-Петербург \
  #datetime.today().year()
]

#pagebreak()

// Start numbering here to skip first page numering
#set page(
  numbering: "1"
)

// To make indent before first header
\
== Задание

Автоматически построить рефераты текстовых документов.

*Ввод*: Массив текстов.

*Вывод*: Массив рефератов.

Максимальный размер каждого из рефератов -- 300 символов (включая пробельные). Если размер реферата превышает указанный порог, то будут оцениваться только первые 300 символов. Тривиальное решение не допускается (первые 300 символов документа).

*Оценка*: ROUGE — близость к набору вручную составленных рефератов на основе N-грамм слов (значение от 0 до 1) + "золотой стандарт".

*Sample Input*:
["Первый текст...", "Второй текст..."]

*Sample Output*:
["Реферат первого текста...", "Реферат второго текста..."]


\
== Выполнение работы

В качесте baseline для задачи автореферирования выбран метод TextRank, основанный на косинусном расстоянии предложений. Все запуски производились на датасете #show link: underline; #link("https://huggingface.co/datasets/IlyaGusev/gazeta")[gazeta].

Последовательнсоть формирования реферата в разработанной программе:
+ Ранжирование по близости предложений к "центральному";
+ Из них выбираются предложения с максимальной полученной метрикой близости, которые в сумме составляют длину менее 300 символов;
+ Внутри выбранных предложений восстанавливается исходный порядок предложений.

Ниже в @td_idf_metrics представлены значения метрики ROUGE.
#figure(
  caption: [
    Результаты измерения метрик на 10000 случайно выбранных из датасета парах text-summary.
  ],
  table(
    columns: (25%, 25%, 25%, 25%),
    [*ROUGE-n*], [*Precision*], [*Recall*], [*F1*],
    "ROUGE-1", $0.1504$, $0.1041$, $0.1127$,
    "ROUGE-2", $0.0415$, $0.0266$, $0.0297$,
    "ROUGE-L", $0.1464$, $0.1008$, $0.1093$,
  ),
) <td_idf_metrics>
