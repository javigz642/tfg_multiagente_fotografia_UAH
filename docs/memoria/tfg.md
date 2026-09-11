# Portada

> La cubierta y la tapa trasera se ajustan al modelo del Anexo I de la
> Normativa de Trabajos Fin de Grado de la Escuela Politécnica Superior
> de la Universidad de Alcalá, aprobada en Junta de Escuela el 25 de
> junio de 2024. El artículo 2.6 de esa normativa fija los datos que
> deben figurar en ella: el escudo de la universidad, el nombre del
> grado, el nombre del centro, el título del trabajo, el nombre del
> estudiante, el del tutor y el del cotutor en su caso, y el curso
> académico.

- Trabajo Fin de Grado

- Grado en Ingeniería de Computadores

- Escuela Politécnica Superior --- Universidad de Alcalá

- Título: Sistema multiagente para el análisis de la composición
  fotográfica

- Autor: Edison Javier Tayan Gualoto

- Tutora: Sira Elena Palazuelos Cagigas

- Curso académico: 2026/2027

# Índice general, índice de figuras e índice de tablas

> Los tres índices se generan de forma automática en el procesador de
> textos a partir de los estilos de título del documento y de los pies
> numerados de tablas y figuras; su contenido no se redacta a mano.

# Resumen

> Los sistemas de evaluación estética automática puntúan una fotografía
> sin explicar por qué. Este trabajo presenta un sistema multiagente
> jerárquico, construido sobre CrewAI, que analiza la composición de una
> fotografía y emite una crítica razonada en la que cada afirmación se
> cita con la métrica que la sostiene. Cuatro especialistas calculan
> diez métricas verificables mediante visión por computador clásica y
> declaran cuándo no son aplicables; un agente crítico las jerarquiza
> según el contexto de la imagen y las verbaliza sin poder alterarlas.
> El sistema prefiere callar a fabricar precisión, y esa disciplina se
> comprueba cita a cita sobre sus informes.

# Abstract

> Automatic aesthetic assessment systems score a photograph without
> explaining why. This work presents a hierarchical multi-agent system,
> built on CrewAI, that analyses photographic composition and produces a
> reasoned critique in which every claim is cited alongside the metric
> that supports it. Four specialist agents compute ten verifiable
> metrics using classical computer vision and declare when those metrics
> do not apply; a critic agent ranks them according to the image context
> and verbalises them without being able to alter them. The system
> prefers to remain silent rather than fabricate precision, and that
> discipline is checked claim by claim against its reports.

# Palabras clave

> Composición fotográfica, sistemas multiagente, visión por computador,
> explicabilidad, modelos de lenguaje, trazabilidad.
>
> *Keywords*: photographic composition, multi-agent systems, computer
> vision, explainability, large language models, traceability.

# Resumen extendido del trabajo

> **Planteamiento del problema.** La evaluación estética automática de
> fotografías ha alcanzado una precisión notable a la hora de predecir
> qué imágenes gustan, pero lo hace produciendo una puntuación sin
> justificación comprensible. Para un fotógrafo que quiere mejorar sus
> tomas, saber que una imagen obtiene un 6,5 sobre 10 no indica qué
> corregir. Los modelos de lenguaje multimodales sí generan
> explicaciones en lenguaje natural, pero un modelo generalista carece
> de la especialización necesaria para tratar por separado las
> dimensiones que componen el encuadre y, sobre todo, no ofrece ninguna
> garantía de que lo que afirma se corresponda con lo que la imagen
> contiene: la explicación puede ser plausible y falsa al mismo tiempo.
>
> **Objetivo e hipótesis.** El objetivo general es desarrollar una
> aplicación en Python que realice un análisis compositivo de
> fotografías digitales mediante un sistema multiagente. La hipótesis de
> trabajo es que separar quién mide de quién explica produce una crítica
> más fiable que la de un único agente que asume a la vez todas las
> dimensiones compositivas: las magnitudes las calcula
> código de visión por computador, auditable y recomputable, y el modelo
> de lenguaje se limita a verbalizarlas citando cada afirmación con la
> métrica que la respalda, con el formato `[métrica = valor]`.
>
> **Arquitectura.** El sistema se organiza en tres niveles y seis
> agentes. El **nivel 1** determina el contexto de la escena con una red
> ResNet18 ajustada por transferencia sobre el conjunto EVA, que
> distingue cinco categorías —animal, arquitectura, paisaje, producto y
> retrato— más una categoría de rechazo, y produce además una
> *percepción compartida* que localiza el sujeto principal combinando un
> detector YOLO con un mapa de saliencia por residuo espectral. El
> **nivel 2** lo forman cuatro especialistas con dimensiones ortogonales
> —composición espacial, líneas y dirección, espacio y aislamiento del
> sujeto, y luz y tono— que calculan en total diez métricas mediante
> técnicas clásicas: detección de bordes de Canny y transformada de
> Hough con estimación del punto de fuga por consenso, segmentación
> GrabCut y varianza del laplaciano, y análisis cromático en CIELAB y
> HSV con agrupamiento circular por *k-means*. El **nivel 3** es un
> agente crítico que arbitra los cuatro informes y redacta la síntesis
> que lee el usuario. Las técnicas de visión se ejecutan en local sobre
> GPU y solo la síntesis emplea una interfaz de programación en la nube,
> lo que mantiene el cálculo verificable fuera del modelo generativo.
>
> **El contrato de confianza por métrica.** La pieza que sostiene el
> resto del diseño es que ninguna métrica se emite como un número
> desnudo: cada una viaja acompañada de una confianza y del motivo que
> la justifica. Cuando las condiciones de aplicabilidad no se cumplen
> —no hay ninguna línea que funcione como horizonte, la imagen es
> acromática, no hay un sujeto acotado que aislar—, la métrica se
> publica igualmente con su valor medido pero con confianza nula, y
> entonces el crítico tiene prohibido citarla. Así, el caso en que una
> regla compositiva no aplica deja de ser un hueco que el modelo
> rellenaría inventando y pasa a ser una afirmación con contenido
> propio. Cada especialista declara además sus condiciones de cierre por
> separado, porque los modos de fallo de cada métrica son distintos.
>
> **Doble autoría.** Cada especialista emite dos cosas con autorías
> distintas: un *informe* de métricas que produce el código y que un
> tercero puede recomputar, y un *diagnóstico* en prosa que redacta el
> modelo de lenguaje citando ese informe. Esa separación permite medir
> por separado la fidelidad de transcripción —si las cifras del texto
> coinciden con las del informe— y la fidelidad métrica-texto —si toda
> afirmación tiene respaldo—, en lugar de tener una única salida opaca.
> El mismo principio se aplica en el nivel 3, donde una herramienta sin
> modelo de lenguaje decide de forma determinista qué métricas son
> citables, con qué cobertura y en qué orden, y solo después interviene
> el modelo para redactar.
>
> **Priorización por contexto.** El orden del discurso lo fija el vector
> de pesos contextuales W, que asigna a cada una de las seis categorías de
> escena una distribución de importancia sobre las cuatro dimensiones,
> derivada de la literatura fotográfica. Ese vector está aislado por
> construcción: no forma parte de la salida del orquestador, de modo que
> es estructuralmente imposible que llegue a los especialistas y
> contamine su medición. 
>
> **Metodología de calibración.** Los umbrales que deciden cuándo una
> métrica es citable no se fijaron por intuición. Cada uno siguió el
> mismo ciclo —contrato provisional, sonda de calibración sobre el
> corpus, revisión visual del autor y contrato cerrado— con el criterio
> explícito de situar el corte en una banda vacía de la distribución
> observada, y con una asimetría deliberada: ante la duda se prefiere
> cerrar de más, porque un falso positivo rompe la verificabilidad
> mientras que un falso negativo solo hace que el sistema calle. La
> revisión visual encontró de forma sistemática lo que el diseño no
> había anticipado, hasta el punto de que varias condiciones previstas
> se descartaron por completo y otras se añadieron después de mirar las
> imágenes.
>
> **Validación y resultados.** El clasificador de contexto alcanza una
> exactitud de 0,8510 sobre las 725 imágenes de prueba, con el error
> concentrado en un par de clases concreto y analizado con Grad-CAM. El
> comportamiento de las condiciones de aplicabilidad se midió sobre un
> corpus de desarrollo de 40 fotografías, y la disciplina de cierre se sostuvo
> sin excepción: sobre las diez síntesis del experimento comparativo,
> ninguna cita el valor de una métrica cerrada, tampoco
> cuando el crítico dispone de una lectura visual de la imagen. Un
> análisis completo tarda algo menos de tres minutos y el coste
> acumulado de la interfaz de programación durante todo el desarrollo
> fue de 14,20 €. El trabajo introduce además una métrica de validación
> propia, la *cobertura de trazabilidad*, que cuantifica qué fracción de
> una crítica descansa en algo verificable.
>
> **Limitaciones.** Los umbrales están calibrados sobre un corpus
> pequeño y propio, de modo que generalizan hasta donde ese corpus sea
> representativo. El detector de objetos hereda las categorías del
> conjunto COCO, lo que deja fuera del aislamiento figura-fondo a
> sujetos como árboles o edificios; es una limitación asumida a
> conciencia y no un error. El trabajo no dictamina calidad artística:
> solo puede concluir si una fotografía se acoge a las convenciones que
> más pesan en su contexto.
>
> **Conclusión.** El resultado no es un sistema que puntúe mejor, sino
> uno que puede ser interrogado: cada frase de la crítica final remite a
> una magnitud que otra persona puede recalcular, y cada silencio tiene
> un motivo declarado.

# Glosario de acrónimos y abreviaturas

> El Anexo II de la normativa de Trabajos Fin de Grado de la Escuela
> Politécnica Superior contempla un único glosario, bajo este nombre
> literal. Reúne, por tanto, dos bloques: las siglas y abreviaturas que
> aparecen en el documento y los términos del dominio fotográfico cuyo
> significado técnico no puede darse por conocido en una titulación de
> ingeniería. Es una consulta rápida y no sustituye a la explicación que
> cada concepto recibe en el capítulo donde se emplea.

**Acrónimos y abreviaturas**

| Sigla | Desarrollo | Uso en este trabajo |
|---|---|---|
| ACM | *Association for Computing Machinery* | Sociedad científica en cuyo congreso ACM Multimedia se publicó el conjunto de datos EVA. |
| API | *Application Programming Interface* (interfaz de programación de aplicaciones) | Vía por la que el sistema invoca los modelos de lenguaje alojados en la nube. |
| AVA | *Aesthetic Visual Analysis* | Conjunto de datos de estética computacional construido con votaciones de una comunidad fotográfica en línea. |
| BLAS | *Basic Linear Algebra Subprograms* | Bibliotecas de álgebra lineal cuyo reparto en hilos explica el no determinismo residual de *k-means*. |
| CIE | *Commission Internationale de l'Éclairage* (Comisión Internacional de la Iluminación) | Organismo que define el espacio de color CIELAB. |
| CIELAB | Espacio de color CIE L\*a\*b\* | Espacio perceptualmente uniforme; su canal de claridad sostiene las métricas de exposición y distribución tonal. |
| CNN | *Convolutional Neural Network* (red neuronal convolucional) | Familia de arquitecturas a la que pertenece el clasificador de contexto. |
| COCO | *Common Objects in Context* | Conjunto de clases con el que se entrenó YOLO; su cobertura semántica limita el aislamiento del sujeto. |
| CSV | *Comma-Separated Values* | Formato de los ficheros de metadatos y de particiones del conjunto de datos EVA. |
| CUDA | *Compute Unified Device Architecture* | Plataforma de cómputo en GPU sobre la que se ejecuta PyTorch. |
| EVA | *Explainable Visual Aesthetics* | Conjunto de datos con el que se ajusta el clasificador de contexto y del que procede la partición de prueba. |
| F1 | Medida F1 | Media armónica de precisión y exhaustividad; se reporta por clase en la evaluación del clasificador. |
| GPU | *Graphics Processing Unit* (unidad de procesamiento gráfico) | Dispositivo local sobre el que corre toda la visión por computador del sistema. |
| Grad-CAM | *Gradient-weighted Class Activation Mapping* | Técnica de explicabilidad que señala las regiones de la imagen que más influyeron en la clase predicha. |
| HSV | *Hue, Saturation, Value* (matiz, saturación, valor) | Espacio de color en el que se analiza la armonía cromática. |
| IA | Inteligencia artificial | Denominación general del campo. |
| IAQA | *Image Aesthetic Quality Assessment* | Evaluación automática de la calidad estética de imágenes; área en la que se inscribe el trabajo. |
| JPEG | *Joint Photographic Experts Group* | Formato de compresión con el que se recodifica la imagen antes de enviarla a la lectura visual. |
| JSON | *JavaScript Object Notation* | Formato en el que se persisten los informes y la crítica de cada ejecución. |
| LIME | *Local Interpretable Model-agnostic Explanations* | Técnica de explicabilidad citada entre las alternativas del estado del arte. |
| LLM | *Large Language Model* (modelo grande de lenguaje) | Modelo generativo que verbaliza las métricas; nunca las calcula. |
| MiDaS | Nombre propio del modelo | Modelo de estimación monocular de profundidad; el anteproyecto lo contemplaba y el trabajo lo descarta por el motivo que expone el apartado 6.4.3. |
| MLLM | *Multimodal Large Language Model* | Modelo grande de lenguaje con entrada visual, citado en el estado del arte. |
| MVP | *Minimum Viable Product* (producto mínimo viable) | Alcance mínimo implementado; delimita lo que queda fuera del sistema entregado. |
| NIMA | *Neural Image Assessment* | Modelo de puntuación estética citado como referencia de la familia que este trabajo no reproduce. |
| ODIN | Nombre propio del método | Método publicado de detección de muestras fuera de distribución, considerado y descartado por alcance. |
| OOD | *Out-of-distribution* (fuera de distribución) | Condición de una imagen ajena a las clases con las que se entrenó el clasificador. |
| OpenMP | *Open Multi-Processing* | Interfaz de paralelismo cuyo reparto en hilos afecta al orden de reducción en coma flotante. |
| PNG | *Portable Network Graphics* | Formato en el que se guardan los paneles de verificación visual. |
| RANSAC | *Random Sample Consensus* | Estimador robusto por consenso, base del cálculo del punto de fuga. |
| ResNet | *Residual Network* (red residual) | Arquitectura convolucional con conexiones de salto; se emplea la variante ResNet18. |
| RGB | *Red, Green, Blue* (rojo, verde y azul) | Espacio de color de partida, sin uniformidad perceptual. |
| RNG | *Random Number Generator* (generador de números pseudoaleatorios) | Fuente de aleatoriedad cuya semilla hay que fijar para que la segmentación sea reproducible. |
| SAM | *Segment Anything Model* | Modelo de segmentación; el anteproyecto lo contemplaba y el trabajo lo descarta por el motivo que expone el apartado 6.4.3. |
| SHAP | *SHapley Additive exPlanations* | Técnica de explicabilidad citada entre las alternativas del estado del arte. |
| ViT | *Vision Transformer* | Arquitectura alternativa a la red convolucional, evaluada y descartada para el clasificador. |
| VLM | *Vision-Language Model* (modelo de visión y lenguaje) | Modelo de lenguaje con entrada visual; sostiene la lectura visual del nivel 3. |
| VRAM | *Video Random Access Memory* | Memoria de la tarjeta gráfica; es una de las restricciones reales del proyecto. |
| WDDM | *Windows Display Driver Model* | Modelo de controlador gráfico que condiciona cómo se mide el consumo de memoria por proceso. |
| XAI | *eXplainable Artificial Intelligence* (inteligencia artificial explicable) | Campo del que procede el planteamiento del problema. |
| YOLO | *You Only Look Once* | Familia de detectores de objetos en una sola pasada; localiza el sujeto principal. |

**Términos del dominio fotográfico**

| Término | Definición |
|---|---|
| Alta clave | Tratamiento en el que la escena se resuelve con tonos predominantemente claros y poco contraste. |
| Armonía cromática | Relación entre los matices dominantes de una fotografía, descrita mediante esquemas convencionales de la teoría del color. |
| Bodegón | Género fotográfico de objetos inanimados dispuestos deliberadamente; en el sistema corresponde al contexto de producto. |
| Encuadre | Límite del fotograma y, por extensión, la decisión de qué fragmento de la escena queda dentro y qué queda fuera. |
| Equilibrio visual | Reparto del peso visual dentro del encuadre; puede ser simétrico o dinámico según se compense la asimetría. |
| Espacio negativo | Fracción del encuadre no ocupada por el sujeto. Su magnitud es una decisión compositiva, no una carencia. |
| Esquema cromático | Etiqueta que resume la relación angular entre los matices dominantes: monocromático (un solo matiz), análogo (matices contiguos), complementario (matices opuestos) u otro. |
| Exposición | Cantidad de luz registrada, descrita aquí por la distribución del canal de claridad de la imagen. |
| Figura y fondo | Distinción entre el sujeto y aquello que lo rodea; su separación es uno de los recursos clásicos para dirigir la mirada. |
| Matiz | Componente del color que indica de qué color es un píxel, con independencia de su saturación y su claridad. Es una magnitud circular. |
| Perspectiva lineal | Efecto por el que las líneas paralelas del espacio tridimensional convergen en la imagen al alejarse de la cámara. |
| Peso visual | Capacidad de un elemento gráfico para atraer la atención y desequilibrar la composición. |
| Profundidad de campo | Franja de la escena que aparece enfocada; su control es uno de los mecanismos de separación entre figura y fondo. |
| Punto de fuga | Punto de la imagen en el que concurren las líneas paralelas del espacio tridimensional bajo perspectiva lineal. Puede caer fuera del fotograma. |
| Recorte tonal (*clipping*) | Pérdida de información en los extremos de la escala de luminancia, cuando las sombras se cierran a negro o las luces se queman a blanco. |
| Regla de los tercios | Convención que divide el encuadre en una cuadrícula de tres por tres y sitúa los elementos principales sobre sus líneas o intersecciones. |
| Saliencia visual | Grado en que una región destaca frente a su entorno y atrae la mirada de forma preatencional, con independencia de qué represente. |

---

**Cuerpo de la memoria**

# Introducción

## Contexto y motivación

> La visión artificial es una disciplina de la inteligencia artificial
> que permite a las máquinas «ver»: extraer información de una imagen
> digital, resolver alguna tarea o interpretar una escena del mundo
> real [FUENTE-AUTOR]. En sus inicios, las tareas de análisis visual
> dependían de enfoques basados en descriptores matemáticos (simetría,
> regla de los tercios, histograma de color, etc.). Sin embargo, estos
> enfoques no son suficientes para capturar la abstracción del arte y de
> las imágenes reales.
>
> Con el avance del *deep learning*, han ido apareciendo arquitecturas
> como las redes neuronales convolucionales (CNN) o los Vision
> Transformers que han demostrado una gran capacidad en tareas de
> clasificación y detección. Ahora bien, estos modelos suelen actuar
> como cajas negras: producen puntuaciones o etiquetas sin ofrecer
> ninguna justificación comprensible para el usuario, lo que limita su
> utilidad práctica en escenarios donde los resultados cualitativos son
> esenciales, como la composición fotográfica.
>
> En la actualidad, áreas como la evaluación de la calidad estética de
> imágenes tienen aplicaciones comerciales que van desde la edición
> automatizada y la mejora de galerías personales hasta la
> recomendación de contenidos en comercio electrónico y la optimización
> de modelos generativos de IA [FUENTE-AUTOR]. A medida que los modelos neuronales
> se vuelven más profundos y complejos, la falta de transparencia se ha
> convertido en una preocupación prioritaria, lo que ha impulsado el desarrollo
> de la inteligencia artificial explicable (XAI) en esta área [FUENTE-AUTOR].

## Planteamiento del problema: la evaluación estética como caja negra

> A pesar del éxito de las redes neuronales profundas para clasificar y
> calificar imágenes, la mayoría de estas herramientas operan como una
> «caja negra». El sistema puede dar una predicción numérica precisa
> sobre el valor estético de una fotografía, pero no explica las causas
> visuales o los criterios de diseño subyacentes que motivaron esa
> respuesta.
>
> En la literatura científica del procesamiento estético de imágenes,
> esta falta de interpretabilidad es un problema ampliamente reconocido.
> Una investigación de Marchesotti et al. [1] sobre el descubrimiento
> automático de atributos estéticos enuncia este inconveniente: «In other
> words, while it is possible to say that an image has a high or low
> aesthetic value, it is impossible to tell why».
>
> Esa carencia limita la utilidad de la IA en la práctica real. Un
> fotógrafo que emplea una herramienta asistida por IA para mejorar sus
> tomas no se beneficia de una simple calificación numérica (como
> 6,5/10). Requiere una retroalimentación estructurada y cualitativa que
> indique si esa cifra se debe a una iluminación defectuosa, un balance
> cromático malo o una composición descentrada. Como expone la
> revisión sistemática sobre IAQA profundo de Daryanavard Chounchenani et al. [2]:
>
> «However, it is important to recognize that these assessments may
> provide aesthetic scores or descriptions, but they do not inherently
> furnish us with the causative factors underlying these aesthetic
> judgments.»
>
> Por ello surge la XAI, cuyo objetivo es dar transparencia a los
> modelos para validar sus lógicas de decisión y fomentar la confianza
> del usuario final:
>
> «The XAI is an emerging field that attempts to break the black-box
> nature of machine learning models. Particularly, the ability to
> explain promotes end-user trust and helps developers to ensure that
> the system is working well or not» [2].

## Hipótesis de trabajo y propuesta de solución

> Frente al problema de la caja negra existen técnicas XAI que informan
> sobre el proceso de decisión:

- Grad-CAM: muestra en qué zonas de la imagen se ha centrado la red.

- Oclusión: elimina partes de la imagen y observa cómo cambia la
  predicción.

- SHAP/LIME: estiman la importancia de diferentes características.

> Y aunque estas técnicas producen explicaciones o aproximaciones al
> comportamiento, no eliminan completamente la caja negra del modelo.
>
> Una posible evolución son los agentes de inteligencia artificial:
> modelos de IA capaces de percibir la información de su entorno, tomar
> decisiones y actuar de forma autónoma para alcanzar un objetivo. La
> diferencia entre un agente de IA y un LLM está en la autonomía operativa:
> mientras un LLM se limita a generar una respuesta a partir de una
> entrada dada, el agente de IA es capaz de encadenar múltiples pasos,
> utilizar herramientas externas y perseguir un objetivo de manera
> sostenida en el tiempo.
>
> No obstante, el empleo de un único agente para resolver tareas que
> exigen diversidad de enfoques presenta limitaciones. Cuando un mismo
> agente debe asumir simultáneamente distintos roles ---el análisis de la
> regla de los tercios, el color de la imagen o la perspectiva---, se
> observa una pérdida de la especialización: el sistema tiende a mezclar
> criterios propios de cada rol, lo que repercute negativamente en la
> calidad y coherencia del resultado final.
>
> Un sistema multiagente resuelve este problema definiéndose como una
> arquitectura en la que múltiples agentes de IA, cada uno especializado en
> una función concreta, cooperan para alcanzar un objetivo común que
> ninguno de ellos resolvería por separado.
>
> Este enfoque introduce elementos que un agente único no
> abarca:

- Especialización de roles: cada agente recibe un objetivo y un entorno
  de actuación acotado, lo que favorece la calidad de su contribución
  específica.

- Comunicación y delegación: los agentes deben intercambiar información
  entre sí, de manera que el resultado de uno pueda ser la entrada de
  otro.

- Orquestación: hace falta un mecanismo que fije el orden de ejecución
  de los agentes, ya sea de manera secuencial, jerárquica o mediante
  otros criterios de coordinación.

> Este paradigma de trabajo colaborativo entre agentes especializados
> constituye el fundamento conceptual sobre el que se ha construido
> CrewAI, un *framework* de código abierto desarrollado en Python. Su
> filosofía de diseño mediante Agent, Task y Crew fue determinante para
> seleccionarlo como base del sistema multiagente que analiza la
> composición de fotografías. En él, cada agente está especializado en
> una dimensión compositiva distinta; un agente orquestador gobierna el
> conjunto y un agente crítico sintetiza la información de los
> especialistas.
>
> La hipótesis de trabajo que recorre el resto de la memoria se enuncia
> así: separar quién mide de quién explica produce una crítica más
> fiable que la de un único agente que asume a la vez todas las
> dimensiones compositivas. El código de visión por computador calcula
> las magnitudes, que son auditables y recomputables por un tercero a
> partir de la misma fotografía, y el modelo de lenguaje no
> las genera: se limita a verbalizarlas citando cada afirmación con la
> métrica que la respalda, con el formato `[métrica = valor]`. De esa
> separación se deriva la pieza que sostiene todo el diseño: ninguna
> métrica se emite como un número desnudo, sino acompañada de una
> confianza que declara cuándo no es aplicable. El apartado 4.4
> desarrolla este principio.

## Objetivos

### Objetivo general

> El objetivo principal de este trabajo es desarrollar una aplicación en
> Python que realice un análisis compositivo de fotografías digitales
> mediante un sistema multiagente. La aplicación usa CrewAI para
> coordinar los agentes especializados, integrando herramientas útiles
> para proporcionar contexto global y métricas detalladas.

### Objetivos específicos

> Los objetivos específicos del trabajo son los siguientes:

- Implementar una arquitectura jerárquica en la que un agente orquestador
  coordine los especialistas en cada dimensión
  compositiva.

- Integrar herramientas del estado del arte ---YOLO para la detección
  de sujetos y algoritmos clásicos de visión por computador (Canny,
  Hough, GrabCut, análisis con CIELAB/HSV)--- para extraer métricas
  cuantitativas verificables, en lugar de los modelos de segmentación y
  de profundidad que el anteproyecto contemplaba.

- Desarrollar métricas de validación objetiva para que cada agente genere
  un informe basado en datos cuantificables. El agente crítico sintetiza
  estos informes priorizando las dimensiones compositivas más relevantes para
  cada tipo de fotografía según un vector de pesos contextuales.

- Garantizar la usabilidad y la visualización de los resultados mediante
  una interfaz gráfica que permita a fotógrafos de distintos niveles
  interactuar con el sistema de manera intuitiva y que muestre junto a
  la crítica la evidencia visual sobre la que se apoya.

## Alcance del trabajo y exclusiones

> El sistema analiza una fotografía cada vez y produce, por cada una,
> una única crítica compositiva; qué entrega exactamente y a quién se
> desarrolla en el apartado 3.1. El trabajo deja fuera de su alcance, de
> forma deliberada, los puntos siguientes:

- No cubre todas las convenciones compositivas: se acota a cuatro
  dimensiones ortogonales y diez métricas, elegidas porque admiten una
  medición verificable y recomputable. Las cuatro preguntas que las
  delimitan están en el apartado 4.2.

- No dictamina calidad artística ni gusto personal, ni es una
  evaluación estética tipo NIMA/AVA: el apartado 3.1 explica en qué se
  diferencia de esas dos familias de sistemas.

- No se despliega en producción pública, por la dependencia de una GPU
  local y por el coste y la exposición de la clave de la API.

- Limitación de cobertura semántica heredada de COCO: los sujetos sin clase
  en el detector (árboles, edificios) quedan fuera del aislamiento
  figura-fondo por diseño, no por error.

- No emplea los modelos de segmentación y de estimación monocular de
  profundidad que contemplaba el anteproyecto: el aislamiento
  figura-fondo se resuelve con algoritmos clásicos, y con ello queda
  fuera la jerarquía de planos que aquel asociaba a esta dimensión
  (apartado 6.4.3).

## Estructura de la memoria

> La memoria se organiza en diez capítulos, agrupados en cuatro bloques:
> el planteamiento (capítulos 1 a 3), la descripción del sistema
> (capítulos 4 a 7), su validación (capítulos 8 y 9) y el cierre
> (capítulo 10).
>
> El capítulo 1 sitúa el trabajo. Describe el problema de la evaluación
> estética automática entendida como caja negra, enuncia la hipótesis de
> partida ---separar quién mide de quién explica---, fija el objetivo
> general y los cuatro objetivos específicos, y delimita el alcance con
> sus exclusiones declaradas.
>
> El capítulo 2 revisa el estado del arte en los frentes que el sistema
> combina: la composición fotográfica clásica y su literatura; la
> evaluación estética automática y los conjuntos de datos que la
> sostienen; las técnicas de visión por computador que producen los
> descriptores compositivos; la clasificación con redes convolucionales,
> el aprendizaje por transferencia y la explicabilidad mediante
> Grad-CAM; los modelos de lenguaje y de visión y lenguaje, con sus
> problemas de alucinación e infidelidad de la explicación; y los
> sistemas multiagente. Cierra situando el trabajo respecto de las tres
> generaciones de intentos de automatizar el juicio estético.
>
> El capítulo 3 fija qué produce el sistema y para quién, declara los
> principios de diseño ---verificabilidad, reproducibilidad,
> explicabilidad, coste y latencia--- y las restricciones reales que
> condicionaron las decisiones técnicas. Describe, además, la metodología
> de calibración que se aplica en todo el trabajo: el ciclo de contrato
> provisional, sonda de calibración, revisión visual y contrato cerrado, y el criterio
> de banda vacía con el que se defiende cada umbral.
>
> El capítulo 4 describe la arquitectura sin entrar todavía en cada
> componente: los tres niveles, los seis agentes y el flujo de datos; la
> separación en cuatro dimensiones compositivas; los contratos de datos
> entre niveles y el aislamiento del contexto de cada tarea; el contrato
> de confianza por métrica, que es la pieza de la que se deriva el resto
> del diseño; la doble autoría de informe y diagnóstico; el vector de
> pesos contextuales W y su aislamiento; la selección de tecnologías; y
> la trazabilidad visual como principio y no como añadido de la
> interfaz.
>
> El capítulo 5 desarrolla el nivel 1. Cubre el clasificador de contexto
> ---selección del conjunto de datos, mapeo de categorías y partición,
> arquitectura, entrenamiento en dos etapas, umbral de rechazo y mapas
> de explicabilidad--- y la percepción compartida, con la detección del
> sujeto, el mapa de saliencia, las tres procedencias posibles del
> cuadro delimitador y el indicador `sujeto_discreto`. Termina con la
> salida estructurada que el nivel 1 entrega al nivel 2 y con la
> evidencia visual que deja por el camino.
>
> El capítulo 6 desarrolla el nivel 2. Documenta primero el patrón común
> a los cuatro especialistas ---herramienta, esquema de doble autoría,
> condición de aplicabilidad, diagnóstico y panel de verificación--- y
> después cada uno de ellos con los mismos seis apartados: qué mide y
> qué no mide, cómo se calcula, qué alternativas se descartaron y con
> qué dato, cómo se calibraron sus condiciones de aplicabilidad, qué se
> midió sobre el corpus de desarrollo y qué no puede afirmar cada
> métrica. Cierra con el diseño de las instrucciones que gobiernan la
> redacción de los diagnósticos.
>
> El capítulo 7 desarrolla el nivel 3 y la entrega al usuario. Explica
> el arbitraje determinista que aplica el vector W sin intervención del
> modelo de lenguaje; la síntesis, con sus cierres y su formato de cita;
> el bloque de tensiones entre dimensiones, que es donde la arquitectura
> multiagente produce algo que ningún especialista aislado podría
> producir; la lectura visual como tercera fuente, con su jurisdicción y
> sus límites; el alcance del juicio que el agente crítico puede emitir;
> y la interfaz que entrega la crítica junto con su evidencia visual.
>
> El capítulo 8 fija el diseño de la validación antes de presentar
> ningún resultado. Delimita qué se puede validar de un juicio estético
> y qué no, distingue los tres conjuntos de imágenes que intervienen y
> el papel de cada uno, describe los instrumentos de medida ---ninguno
> de los cuales emplea un modelo de lenguaje para juzgar a otro--- y
> enuncia los cuatro protocolos experimentales.
>
> El capítulo 9 presenta los resultados en el mismo orden: la evaluación
> del clasificador de contexto con su matriz de confusión y el análisis
> del error dominante; la aplicabilidad de las condiciones de los
> especialistas, la fidelidad entre métrica y texto y la comparación del
> agente crítico con y sin lectura visual; y el coste, la latencia y el
> consumo de recursos. Termina distinguiendo qué demuestran esos
> resultados y qué no, y enumerando las limitaciones del trabajo.
>
> El capítulo 10 cierra con el grado de cumplimiento de los objetivos,
> las aportaciones del trabajo, las lecciones aprendidas durante el
> desarrollo y las líneas de trabajo futuro.

# Estado del arte y fundamentos teóricos

## Composición fotográfica: reglas clásicas y su literatura

> El encuadre constituye, para Michael Freeman, el límite activo sobre el
> que se construye toda composición fotográfica, y no simplemente un
> borde que recorta la escena [3]. A diferencia de la pintura
> donde la imagen se construye de forma acumulativa sobre un lienzo en
> blanco, la fotografía es un proceso sustractivo: el fotógrafo
> selecciona y delimita un fragmento de la realidad que luego interactúa
> con los bordes del fotograma. Dentro de ese límite, cualquier sistema
> de subdivisión altera de inmediato la dinámica visual interna de la
> imagen. El más extendido en la literatura y en la práctica fotográfica
> es la regla de los tercios: dividir el encuadre mediante una
> cuadrícula imaginaria de 3x3, de modo que los elementos principales
> estén sobre las líneas o en sus intersecciones. Esta ubicación
> descentrada evita la rigidez del centrado absoluto y produce un
> equilibrio dinámico que esta tradición compositiva considera más
> agradable a la vista que la simetría estricta [3].
>
> El equilibrio visual es, junto con el encuadre, el eje estructural de
> la composición clásica. Freeman lo describe recurriendo a la analogía
> de una balanza: cualquier elemento gráfico (una masa oscura, contraste
> de color, agrupación de puntos) introduce un peso visual que
> desequilibra la imagen y genera en el espectador la necesidad de
> compensarlo [3]. El equilibrio puede clasificarse en dos
> modos de organización:

- Equilibrio simétrico o estático que sitúa a los elementos a distancias
  equivalentes respecto al centro geométrico y produce una sensación de
  solemnidad y estabilidad a costa de resultar, según esta misma
  literatura, menos dinámico para el espectador.

- Equilibrio dinámico, en cambio, asume la asimetría como recurso: un
  elemento de peso visual reducido puede compensar a un sujeto dominante
  si se sitúa en el extremo opuesto del encuadre. Präkel formaliza esta
  idea como la distribución del peso visual a lo largo del encuadre, de
  manera que el ojo del observador encuentre un centro de gravedad
  visual satisfactorio [4].

> Las líneas son el elemento con mayor capacidad para dirigir la mirada
> e introducir dinamismo, y su efecto depende de la orientación. Las líneas
> horizontales se asocian instintivamente con el horizonte y con el suelo
> como base estable, transmiten reposo y tranquilidad. Las líneas
> verticales expresan fuerza de la gravedad y se relacionan con
> elementos como la figura humana o edificios, aportando sensación de
> altura. Las curvas introducen un cambio progresivo de dirección que se
> percibe como fluido y elegante. Las diagonales son el elemento más
> dinámico de la composición clásica: encierran una tensión no resuelta
> entre lo horizontal y lo vertical. Además, por efecto de la
> perspectiva lineal, las líneas paralelas que se alejan de la cámara
> convergen visualmente hacia uno o más puntos de fuga, lo que aporta al
> encuadre una ilusión de profundidad y distancia [3].
>
> Esa misma convergencia es, para Präkel, el recurso compositivo central
> en escenas lineales, como la fotografía de arquitectura y las escenas
> urbanas [4]: organiza la mirada del espectador a lo largo
> del eje de profundidad y refuerza la sensación de espacio
> tridimensional. Präkel identifica además la separación entre la figura
> y fondo como un recurso para guiar la atención del espectador y
> establecer una jerarquía visual en la imagen, y describe tres
> mecanismos con los que el fotógrafo puede lograrla: el desenfoque de
> fondo mediante el control de la profundidad de campo, el contraste
> cromático entre el sujeto y su entorno y el contraste de textura entre
> ambos.
>
> Präkel también menciona que rara vez una imagen utiliza
> exclusivamente uno de los elementos formales de la composición: las
> mejores son una combinación de ingredientes, y analizar cómo se
> incorpora cada uno ayuda a comprender la composición. En el arte esos
> elementos pueden ser la línea, la forma, el volumen, el espacio, la
> textura, la luz y el color.
>
> El último de esos elementos ilustra bien por qué la composición no se
> agota en la geometría del encuadre: la presencia del color, y también
> su ausencia, son decisiones del fotógrafo antes que accidentes
> técnicos.
>
> «Cuando uno fotografía personas en color, fotografía sus ropas. Cuando
> fotografía personas en blanco y negro, fotografía su alma». Ted Grant
> (fotógrafo canadiense) [FUENTE-AUTOR]

## Evaluación estética automática: AVA y EVA

> AVA (Aesthetic Visual Analysis), introducido por Murray et al. [5], es
> una base de datos masiva diseñada para el estudio de la
> estética computacional.

- Tamaño y origen: Contiene aproximadamente 255 000 imágenes asociadas a
  un conjunto de 963 desafíos temáticos organizados por la comunidad en
  línea de fotografía [www.dpchallenge.com](http://www.dpchallenge.com)

- Construcción de las puntuaciones: Las valoraciones proceden de las
  votaciones de los propios miembros del sitio web. Cada
  fotografía fue calificada de forma independiente por un rango de 78 a
  549 votantes (con un promedio de 210 votos por imagen) utilizando una
  escala discreta del 1 al 10. El conjunto de datos proporciona la distribución
  completa de las puntuaciones para cada imagen.

- Uso: Se emplea como el *benchmark* de referencia estándar para el
  entrenamiento y la evaluación de modelos de calidad estética.

> **EVA (Explainable Visual Aesthetics): innovación frente a AVA.** El
> conjunto de datos EVA propuesto por Kang et al. [6] surge como una
> respuesta directa a la opacidad de los conjuntos de datos tradicionales como
> AVA, cuyas puntuaciones de concursos suelen estar sesgadas por el tema
> específico de cada desafío y carecen de criterios unificados de
> evaluación. Los datos se obtuvieron mediante una plataforma web donde
> los participantes evaluaban directamente cada imagen respondiendo
> preguntas. EVA aporta etiquetas multidimensionales y explícitas como
> el puntuaje general, la dificultad subjetiva del juicio, la magnitud
> de atributos estéticos y la importancia relativa declarada que los
> evaluadores otorgaron a cada atributo al emitir su voto [6]. Las fotos
> se dividieron en seis categorías con una decisión
> metodológica clave.

- Control de sesgo por contenido (Equilibrio de probabilidad): La
  literatura científica demuestra que el contenido semántico influye en
  cómo las personas perciben la estética. Si un conjunto de datos se evaluara sin
  filtros, sesgaría las votaciones de los usuarios. Al estructurarlo en
  clases, los investigadores pudieron diseñar el experimento controlado
  de manera que cada participante viera diferentes categorías de la
  imagen con una probabilidad similar.

- Variación de la importancia de los atributos: Es uno de los hallazgos
  centrales de EVA. Al segmentar los análisis
  por clases, los autores observaron que la importancia de cada
  atributo estético varía de forma acusada en función de la temática de la
  foto. En paisajes y escenas naturales, los atributos fotográficos y
  perceptuales como la luz, color o profundidad adquieren un peso
  dominante ya que la variedad de tonos y atmósfera suelen definir la
  belleza de toma. En arquitectura y escenas urbanas, las líneas
  geométricas pasan a tener una prioridad mayor para el observador.

- Categorías de EVA: EVA dividió las imágenes en seis clases: animales,
  arquitectura/escenas urbanas, humanos, escenas naturales/rurales,
  bodegón y otros. Esto se hizo con dos fines: el
  primero es garantizar que los evaluadores vieran diferentes tipos de
  fotografía con una probabilidad similar, equilibrando el sesgo de
  contenido en el juicio; y segundo, analizar cómo varía la importancia
  de cada atributo estético según el género de la imagen. Las primeras 5
  categorías las tomaron de un trabajo de Liu y Wang, y las fotografías
  ambiguas o con otra categoría van a la clase de «otros».

## Visión por computador para descriptores compositivos

> Frente a extraer descriptores mediante una red neuronal aprendida,
> este trabajo recurre a un conjunto de técnicas clásicas de visión por
> computador, la mayoría anteriores a la generalización del aprendizaje
> profundo, para obtener las magnitudes geométricas, de segmentación y
> cromáticas que consumen los cuatro especialistas del nivel 2. La
> elección no es una limitación técnica, sino una condición de diseño:
> son las reglas compositivas más separables y útiles en el campo de la
> ingeniería y la visión artificial. Estas técnicas producen una salida
> numérica, con una formulación matemática y recomputable por un tercero
> a partir de la misma imagen. Los tres bloques siguientes repasan los
> fundamentos en los que se apoyan la geometría de las líneas (apartado 2.3.1),
> la saliencia y la segmentación figura-fondo (apartado 2.3.2) y el análisis del
> color (apartado 2.3.3).

### Detección de bordes y transformada de Hough; estimación del punto de fuga

> La extracción de estructura lineal empieza con el detector de bordes
> de Canny [7], que plantea la detección como un problema de
> optimización sobre tres criterios (buena detección, buena localización
> y respuesta única por borde). A partir de ellos surge el operador de
> Canny que se aplica en cuatro pasos:

a.  Suavizado de la imagen original con filtro gaussiano para eliminar
    el ruido.

b.  Obtención del gradiente en cada píxel.

c.  Adelgazamiento del ancho de bordes, hasta lograr bordes de un píxel
    de ancho (supresión no máxima).

d.  Umbralización por histéresis fijando dos umbrales para localizar
    mejor los bordes.

> Cuatro décadas después de su publicación sigue siendo uno de los
> detectores de bordes de referencia en visión por computador clásica.
>
> Sobre el mapa de bordes, la transformada de Hough resuelve la
> detección de rectas convirtiendo el problema en una votación en el
> espacio de parámetros. En él, cada punto de borde vota por todas las
> rectas que pasan por él, y se obtienen las rectas reales que más
> puntos han acumulado. Su variante probabilística, propuesta por Matas et
> al. [8], evita procesar todos los puntos de borde, ya que
> muestrea aleatoriamente hasta un cierto criterio de parada. A
> diferencia de la transformada clásica, devuelve directamente los
> extremos de cada segmento y no solo su recta soporte. Esa es la
> propiedad que la hace útil aquí, porque la longitud del segmento es en
> sí misma una magnitud de interés.
>
> La estimación del punto de fuga explota que, bajo proyección en
> perspectiva, las líneas paralelas del mundo real convergen en un punto
> común de la imagen. Se debe buscar el punto que maximiza el consenso
> entre las intersecciones de los segmentos detectados, y RANSAC [9] es el
> algoritmo estándar: ajusta el modelo
> a subconjuntos mínimos de datos y cuenta cuántas observaciones son
> compatibles con cada hipótesis, lo que lo hace robusto frente a
> segmentos que no pertenecen a ninguna familia de líneas convergentes.
> Un punto de fuga existe para cualquier conjunto de rectas, converjan o
> no en la realidad. Hay que decidir, por tanto, si la imagen tiene una
> perspectiva real o solo líneas caóticas sin estructura común. De ahí
> que la estimación deba contrastarse frente a un modelo de referencia
> ---líneas de orientación aleatoria--- antes de aceptarla como
> significativa, como se detalla en el apartado 6.3.2.

### Saliencia y segmentación figura-fondo

> La saliencia visual describe el grado en que una región destaca frente
> a su entorno y, por tanto, tiene capacidad para atraer la atención de
> forma preatencional. No equivale a reconocer un objeto: una zona puede
> resultar saliente por su contraste de luminancia, color, orientación o
> textura sin que el sistema conozca su identidad semántica. Esta
> distinción resulta útil para el análisis compositivo, porque permite
> aproximar dónde se concentra el peso visual de la imagen incluso en
> escenas en las que un detector de objetos no encuentra una clase
> conocida.
>
> Para obtener esta información se emplea el método de residuo espectral
> de Hou y Zhang [10]. El algoritmo transforma la imagen al dominio de
> Fourier y separa el espectro en amplitud y fase. Sobre el logaritmo de
> la amplitud calcula una versión suavizada que representa la estructura
> espectral esperable; la diferencia entre ambas constituye el residuo
> espectral. Al reconstruir la señal en el dominio espacial conservando
> la fase se obtiene un mapa continuo en el que los valores altos indican
> regiones que se apartan de las regularidades globales de la escena. Es
> un procedimiento rápido, independiente de categorías aprendidas y
> adecuado para producir una magnitud espacial reproducible, aunque su
> salida señala contraste o rareza visual y no garantiza por sí misma la
> existencia de un sujeto.
>
> Para convertir ese mapa continuo en regiones se recurre al método de
> Otsu [11]. Este selecciona automáticamente el umbral que maximiza la
> varianza entre dos clases del histograma —píxeles de baja y alta
> saliencia—, evitando fijar un valor absoluto común para fotografías
> con distribuciones muy diferentes. La binarización permite extraer
> componentes conexas, contornos, áreas y rectángulos envolventes. Sin
> embargo, el umbral siempre produce una partición cuando existe
> variación suficiente en el mapa; por ello, la mayor región conexa no debe
> interpretarse automáticamente como un objeto real. Su tamaño y su
> coherencia deben someterse después a criterios de plausibilidad.
>
> Cuando se dispone de un rectángulo fiable alrededor del sujeto, la
> separación entre figura y fondo puede refinarse mediante GrabCut
> [12]. Este algoritmo inicializa modelos
> de color para primer plano y fondo mediante mezclas de gaussianas y
> representa la imagen como un grafo. Los píxeles forman los nodos y las
> aristas combinan dos tipos de información: la compatibilidad de cada
> píxel con los modelos de color y la continuidad entre píxeles vecinos.
> Un corte mínimo del grafo obtiene la máscara, tras lo cual los modelos
> se reestiman y el proceso se repite de forma iterativa. Así, el
> rectángulo inicial actúa como una condición aproximada y no como el
> contorno definitivo del sujeto.
>
> La máscara resultante hace posible medir propiedades compositivas que
> no pueden obtenerse únicamente a partir de un rectángulo, como la
> proporción de espacio negativo o la diferencia de densidad de detalle
> entre figura y fondo. Esta última puede estimarse aplicando el operador
> Laplaciano y comparando la varianza de su respuesta en ambas regiones:
> una varianza mayor indica más cambios locales de intensidad. Debe
> interpretarse como detalle espacial y no como una medición inequívoca
> del enfoque o de la profundidad de campo, pues una superficie lisa
> puede producir una respuesta baja aun estando correctamente enfocada.
> En consecuencia, tanto la saliencia como GrabCut proporcionan
> descriptores geométricos auditables, pero necesitan condiciones
> explícitas de aplicabilidad para no confundir una partición algorítmica
> con una identificación semántica fiable.

### Espacios de color y agrupamiento cromático (CIELAB/HSV, k-means)

> El análisis tonal se apoya en el espacio CIELAB, definido por la
> Comisión Internacional de Iluminación en 1976 con el objetivo
> explícito de aproximar la uniformidad perceptual: una misma distancia
> numérica entre dos puntos del espacio debe corresponder, más o menos,
> a una misma diferencia de color percibida por un observador humano,
> algo que RGB no garantiza. Su canal L\* codifica la claridad entre 0
> (negro) y 100 (blanco). El canal a\* representa la componente
> cromática en un eje que enfrenta los colores rojo y verde. El canal
> b\* complementa la información en un segundo eje que enfrenta los
> colores amarillo y azul.
>
> El análisis del esquema cromático se apoya en cambio en el espacio HSV
> [13], que reformula el cubo RGB en coordenadas cilíndricas de
> matiz, saturación y valor. Su interés aquí es que aísla el matiz (de
> qué color es un píxel, independientemente de lo saturado o luminoso
> que sea) en un único ángulo que permite tratar las relaciones
> cromáticas (monocromía, analogía, complementariedad) como relaciones
> angulares, como las describe la teoría clásica del color en la rueda
> cromática.
>
> Para reducir la nube de matices de una imagen a un pequeño número de
> colores dominantes se recurre a *k-means*, el algoritmo de agrupamiento
> particional que minimiza la varianza dentro de cada grupo asignando a
> cada punto su centroide más cercano y recalculando después los
> centroides. El matiz es una magnitud circular, por lo que 0º y 360º son
> el mismo color, y agruparlo en una recta real puede partir en dos un
> color que en realidad es uno solo, por lo que el ángulo se proyecta
> sobre la circunferencia unidad antes de agrupar.

## Clasificación con CNN: ResNet, transfer learning y Grad-CAM, CNN frente a Vision Transformers

> **Redes convolucionales.** Una red neuronal convolucional (CNN) explota
> dos propiedades estructurales de las imágenes que una red totalmente
> conectada ignora: la localidad (un píxel se relaciona sobre todo con
> su entorno inmediato) y la equivarianza a la traslación (un borde o
> una textura se reconoce igual esté donde esté en el encuadre). Los filtros
> convolucionales comparten pesos a lo largo de toda la imagen, lo que
> reduce el número de parámetros frente a una capa densa y actúa como un
> sesgo inductivo: la red parte ya sabiendo que la imagen tiene
> estructura espacial local, en vez de tener que aprenderlo desde cero a
> partir de datos.
>
> **ResNet y el aprendizaje residual.** Apilar más capas convolucionales
> debería mejorar la capacidad de representación, pero He et al. [14]
> mostraron que a partir de cierta profundidad, las redes sin conexiones
> adicionales degradan su precisión incluso en entrenamiento; no es un
> sobreajuste, sino un problema de optimización. Los gradientes cada vez
> más difíciles de propagar hacen que las capas adicionales dejen de
> aprender nada útil. Como solución, ResNet introduce conexiones de
> salto que suman la entrada de un bloque a su salida, de modo que cada
> bloque no tiene que aprender la transformación completa, sino solo el
> residuo. Esto facilita que el gradiente fluya directamente hacia capas
> tempranas y permitió entrenar redes de cientos de capas. La familia
> ResNet ofrece variantes de distinta profundidad, y el sistema de este
> proyecto incorpora ResNet18, la más ligera. Toma el papel de
> clasificador de contexto, el primer componente del nivel 1, y no
> resuelve una tarea de grano fino: solo clasifica la escena en las 5
> categorías.
>
> **_Transfer learning_.** Entrenar una CNN de esta profundidad desde cero
> exige conjuntos de datos del orden de millones de imágenes. EVA incluye
> fotografías etiquetadas por contexto, aunque su número puede ser muy
> pequeño para una CNN desde cero. La solución
> estándar es el *transfer learning*: aquí se toma ResNet18 preentrenada en
> ImageNet con 1,2M de imágenes y 1000 clases. Se pueden reutilizar las
> representaciones visuales de bajo y medio nivel que ya ha aprendido
> (bordes, texturas, formas), que son independientes de la tarea final.
> Sobre esa base se sustituye la última capa totalmente conectada (de
> 512 a las 5 clases de contexto destacables de EVA: animal,
> arquitectura, paisaje, producto/bodegón, retrato/humano) y se
> reentrena (ajuste fino). No se reentrena la red entera con el mismo
> esfuerzo que el preentrenamiento original, sino que se afina un modelo
> que ya sabe ver.
>
> **Grad-CAM.** Un clasificador que solo devuelve una etiqueta y una
> probabilidad softmax sigue siendo una caja negra, ya que no dice en
> qué se ha fijado la red para decidir. Grad-CAM [15] resuelve esto sin
> modificar la arquitectura ni reentrenar nada:
> calcula el gradiente de la puntuación de la clase predicha respecto a
> los mapas de activación de una capa convolucional (usualmente la
> última o la que conserva más semántica de alto nivel sin haber perdido
> aún la disposición espacial). Promedia esos gradientes por canal para
> obtener un peso de importancia y combina esos pesos con los propios
> mapas de activación, con lo que produce un mapa de calor que señala
> qué regiones de la imagen más contribuyeron a la predicción. En el
> sistema, se aplica sobre el último bloque residual de
> ResNet18 (layer4\[-1\]) y genera una superposición que se guarda junto
> a cada clasificación (salvo cuando la imagen se rechaza como «otro»
> porque entonces no hay una clase objetivo que explicar). Es la pieza
> que conecta el único componente de *deep learning* puro del sistema con
> el mismo principio de verificabilidad que rige todo lo demás: no basta
> con la etiqueta, tiene que poder mostrarse la evidencia.
>
> **CNN frente a Vision Transformers.** Los Vision Transformers (ViT) trasladan
> la arquitectura Transformer a imágenes dividiéndolas en parches fijos,
> que se embeben linealmente y se procesan como una secuencia mediante
> *self-attention* [16]. A diferencia de una CNN, un
> ViT no incorpora ningún sesgo de localidad: cada parche puede atender
> a cualquier otro desde la primera capa. Esto es una ventaja cuando hay
> datos suficientes para que el modelo aprenda esas relaciones
> espaciales por sí mismo: en su artículo original, preentrena sobre un
> conjunto de datos de aproximadamente 300M imágenes. Pero a su vez, sin
> preentrenamiento a esa escala, un ViT tiende a rendir peor que una CNN
> comparable sobre conjuntos de datos medianos o pequeños. A esto se le suma un
> coste estructural: la complejidad del *self-attention* crece de forma
> cuadrática con el número de parches (y por tanto con la resolución de
> la imagen), frente al coste lineal de una convolución.
>
> Estas propiedades de datos y de coste estructural son las que llevan a
> elegir una CNN para el clasificador de contexto del sistema. El
> ajuste fino se hace sobre el conjunto de datos EVA muy por debajo de los
> volúmenes que un ViT demanda. Además, el clasificador comparte
> GPU con los modelos de nivel 2 (YOLO y el resto de especialistas que
> corren en local) y debe devolver su etiqueta con la latencia mínima
> posible antes de que estos especialistas arranquen: cargar un ViT
> ocuparía VRAM que esos modelos necesitan. No se busca rendimiento de
> última generación en clasificación de imagen general, sino resolver
> correctamente un clasificador de contextos (5 clases + rechazo por
> umbral de confianza) con el menor coste posible, que es precisamente
> el régimen en el que la literatura revisada no atribuye ventaja a un
> ViT frente a una CNN pequeña como ResNet18. La comparación no se ha
> ejecutado en este trabajo: la elección se apoya en las propiedades de
> datos y de coste descritas, no en una medición propia.

## Modelos de lenguaje y VLM: capacidad explicativa, alucinación e infidelidad de la explicación.

> Los grandes modelos de lenguaje (LLM) y sus variantes con entrada
> visual (VLM) han hecho posible una forma de interacción antes
> inalcanzable: describir una fotografía, razonar sobre su composición y
> producir una crítica en lenguaje natural a partir de una simple instrucción.
> Esta capacidad puede ser un problema si se usa sin control, y es el
> motivo por el que este trabajo no delega en un LLM ni la medición ni
> el juicio, sino la verbalización de evidencia ya calculada. Esto exige
> revisar antes qué puede y qué no puede garantizar un modelo generativo
> cuando se le pide que explique.
>
> Un LLM/VLM generalista produce una respuesta fluida ante prácticamente
> cualquier entrada aunque no tenga evidencia suficiente para
> sostenerla. Este fenómeno se conoce como alucinación y, pese a los
> avances en reducirlo, sigue siendo un problema real. En el ámbito
> multimodal, el riesgo se agrava, porque
> el modelo puede describir con detalle un objeto, color o relación
> espacial que la imagen no contiene, sin que exista ninguna señal
> explícita de incertidumbre en el texto que lo delate. Este trabajo
> cuenta con una observación propia del fenómeno, no solo con la
> referencia bibliográfica: privado de la herramienta que le suministra el
> vector de pesos contextuales, el agente crítico devolvió una fila de
> valores plausible pero incorrecta, sin que el texto permitiera
> distinguirla de un dato real. La ocurrencia y su alcance ---una
> comprobación sobre fragmentos de la plantilla, no un barrido--- están en
> el apartado 7.2.5.
> Esto se une con la infidelidad de las explicaciones, en la que un texto
> puede ser coherente y convincente sin reflejar el proceso
> que realmente produjo la respuesta. Así, un VLM puede justificar una
> composición mediante la regla de los tercios sin haber realizado
> ninguna medición real de la fotografía.
>
> Sin embargo, el problema no es simplemente utilizar un LLM con
> imágenes, sino hacerlo sin una fuente verificable detrás de sus
> afirmaciones. Una comprobación exploratoria propia apunta en esa
> dirección: al exigir a un modelo generalista ejecutar
> código con OpenCV/NumPy para obtener cada cifra, las métricas
> producidas eran consistentes y trazables. No es una medición con
> protocolo, así que se recoge como indicio de que las herramientas
> pueden reducir la alucinación y no como demostración. Sus limitaciones
> son además claras: el código se genera de manera específica para ello,
> no existe una calibración previa ni tasas de error conocidas, el
> proceso puede variar entre ejecuciones y el coste es mayor que el de
> un flujo de procesamiento local fijo.
>
> Por estas razones, el sistema opta por separar las funciones: los
> modelos de lenguaje no calculan ni deciden, sino que verbalizan
> resultados obtenidos mediante métricas verificables. Del mismo modo,
> se utiliza una CNN local para la clasificación de contexto en vez de
> delegar el trabajo a una VLM por API, ya que permite un proceso de
> entrenamiento y evaluación controlado, menor dependencia de servicios
> externos y técnicas de explicabilidad como Grad-CAM.

## Sistemas multiagente: patrones de coordinación y orquestación jerárquica. CrewAI.

> Un sistema multiagente basado en modelos de lenguaje (LLM) descompone
> una tarea compleja en unidades más pequeñas, cada una resuelta por un
> agente con un rol, un objetivo y unas herramientas delimitadas, en
> lugar de confiar la tarea a un único modelo generalista que razona
> sobre todo el problema a la vez [17]. La descomposición
> no es solo organizativa, sino que reduce el contexto que debe sostener
> cada llamada al modelo y permite que cada agente se especialice en una
> subtarea en vez de comprimir todo el razonamiento en una única plantilla de instrucciones
> larga.
>
> Respecto a su proceso de ejecución, se ha optado por uno secuencial,
> aunque el sistema se estructura como una arquitectura jerárquica
> debido a su clasificación en tres niveles. La alternativa era
> `Process.hierarchical`, donde un LLM gestor reparte el trabajo entre
> los agentes disponibles en cada momento. Son arquitecturas de tipo
> *puppeteer*, en las que una política central decide en cada paso qué
> agente razona, y la literatura sobre orquestación de sistemas
> multiagentes en producción las señala como un riesgo de gobernanza:
> hacen falta mecanismos de observabilidad añadidos aparte para sostener
> la transparencia del reparto [18]. La forma
> de jerarquía es incompatible con las técnicas
> de explicabilidad que quiere dar este trabajo. Si un LLM decide qué
> agente se ejecuta y cuál no, esa decisión de activación no tiene una
> regla auditable detrás y no es posible reconstruir a partir de la
> salida, por qué un especialista no llegó a intervenir.
>
> Por ello, el sistema usa `Process.sequential`: los cuatro especialistas
> de nivel 2 se ejecutan siempre y la pertenencia al flujo de datos
> queda fijada en el diseño, no en tiempo de ejecución por ningún LLM.
> Para abordar el problema de qué métrica en cada agente compositivo es
> citable, cada una lleva asociado un campo de confianza, recomputable
> por un tercero y no fijado por un juicio del modelo de lenguaje. El
> sistema no responde a ¿qué agente merece
> intervenir?, sino a ¿qué mide este agente que sea fiable en esta
> imagen en concreto?
>
> CrewAI formaliza esto con tres abstracciones: Agent(rol, objetivo,
> herramientas), Task(description, expected_output, agent) y Crew (el
> conjunto orquestado).

## Posicionamiento del trabajo respecto al estado del arte

> Los apartados anteriores describen tres generaciones de intentos de
> automatizar el juicio estético, y las tres comparten el mismo punto
> ciego. La primera ---reglas fotográficas clásicas formalizadas como
> descriptores hechos a mano (apartados 2.1 y 2.3)--- es interpretable por
> construcción pero frágil: cada regla se calcula de forma aislada y
> nadie arbitra entre ellas cuando entran en conflicto. La segunda
> ---modelos de puntuación entrenados sobre AVA, EVA o el propio NIMA [19]
> (apartado 2.2)--- resuelve la fragilidad con aprendizaje de extremo a extremo,
> pero al precio de convertirse en caja negra: produce un número de
> calidad sin ningún mecanismo para preguntarle por qué. Marchesotti,
> Murray y Perronnin [1] nombraron esta tensión con precisión
> ---«current computational approaches to aesthetic image analysis
> either provide accurate or interpretable results»--- y su propia
> respuesta, aprender atributos nombrables como capa intermedia
> (retomada por Kong et al. [20], con adaptación al contenido), sigue
> siendo estadística: el atributo se infiere, no se mide. La línea
> continúa hoy con marcos de conceptos aprendidos como el de Liu y
> Wagemans [21], que sitúan la predicción sobre un subespacio de
> conceptos «comprensibles por humanos» --- un paso más hacia la
> interpretabilidad, pero sigue siendo una explicación post-hoc de una
> decisión tomada dentro de una red, no una decisión construida desde
> piezas verificables.
>
> La tercera generación, la más reciente, sustituye el atributo
> aprendido por un modelo multimodal que critica la imagen en lenguaje
> natural. PhotoEye [22] entrena un MLLM sobre miles de
> discusiones reales entre fotógrafos para producir crítica fluida y
> superior a la de otros modelos generalistas; Abe et al. [23] llegan
> más lejos y proponen un sistema de dos agentes LLM
> ---uno que entrevista al usuario, otro que extrae rasgos semánticos---
> que predice la preferencia estética individual con menos error que el
> propio ser humano evaluándose a sí mismo, tiempo después. Son
> resultados fuertes, pero ninguno de los dos sistemas puede señalar,
> dentro de su propia crítica, qué frase está sostenida por un cálculo
> reproducible y cuál es una construcción verbal del modelo. La revisión
> más reciente sobre evaluación estética con aprendizaje profundo
> [2] lo deja como reto abierto
> explícito: «the need for explainable AI to understand the causative
> factors behind aesthetic judgments».
>
> Este trabajo no compite en ese terreno: no entrena un modelo para
> predecir mejor una puntuación humana, y de hecho se niega
> deliberadamente a producir esa puntuación (apartado 1.5). Ocupa un hueco
> distinto y más estrecho, en la intersección de dos líneas que la
> literatura revisada mantiene separadas: los descriptores clásicos
> auditables del apartado 2.3 (Hough, saliencia, CIELAB, *k-means*) como única
> fuente de cómputo, y el LLM reducido a una función de dos tareas
> ---verbalizar citando su fuente y arbitrar según un vector de pesos
> explícito y versionado--- nunca a una función de cálculo. La
> arquitectura multiagente (apartado 2.6) tampoco persigue el mismo objetivo que
> Abe et al. [23]: no reparte el trabajo entre agentes para mejorar
> una predicción, sino para mantener separadas y auditables cuatro
> dimensiones compositivas que una única plantilla de instrucciones monolítica fusionaría sin
> poder deshacer después.

# Planteamiento del sistema y metodología de trabajo

El apartado 2.7 sitúa el hueco que este trabajo ocupa entre los descriptores
clásicos auditables y la crítica generada por un modelo de lenguaje. Este
capítulo lo convierte en un encargo concreto antes de describir ninguna
arquitectura: qué produce el sistema y para quién, bajo qué principios y con
qué restricciones reales se tomaron las decisiones técnicas, y con qué método
se fijó cada uno de los umbrales que deciden si una métrica puede citarse.

## Qué produce el sistema y para quién

> El sistema recibe una imagen, clasifica su contexto en cinco
> categorías (animal, arquitectura, retrato, paisaje, producto) o en la
> categoría «otro», y produce una crítica compositiva
> estructurada: un texto de síntesis en prosa, acompañado de un conjunto
> de afirmaciones donde cada una cita explícitamente la métrica
> cuantitativa que la sostiene (afirmación `[métrica = valor]`), más las
> salvedades de fiabilidad sobre lo que la fotografía concreta no permite afirmar y
> seis paneles visuales de verificación, uno por etapa de análisis, que
> muestran sobre la propia imagen qué ha medido cada componente.
>
> Esto lo distingue de dos familias de sistemas con las que podría
> confundirse. Frente a los modelos de evaluación estética automática
> (AVA, EVA, NIMA), que devuelven una puntuación numérica sin
> razonamiento verbal, este sistema no puntúa: produce una crítica
> razonada en lenguaje natural. Y frente a un modelo de lenguaje
> generalista con capacidad multimodal al que se le pide «evalúame esta
> foto», su salida es plausible pero no verificable. No hay forma de
> saber si el sujeto está realmente en el tercio izquierdo o si el LLM
> lo ha deducido del contexto. En cambio, cada afirmación de contenido
> cuantitativo de este sistema remite a un número calculado por un
> algoritmo determinista y auditable por un tercero sin volver a invocar
> ningún modelo de lenguaje.
>
> El destinatario es un fotógrafo aficionado o semiprofesional que
> quiere entender por qué una composición funciona o no. No basta con
> responder que la foto es buena: el sistema debe describir el grado de
> conformidad de una fotografía con un conjunto de convenciones
> compositivas clásicas (regla de los tercios, líneas de fuga,
> equilibrio visual, armonía cromática...), ponderadas según el género
> fotográfico detectado. No dictamina calidad artística ni sustituye el
> criterio del autor de la imagen.
>
> En la práctica, el sistema se entrega como una interfaz web, el
> usuario sube una fotografía, espera mientras las seis tareas se
> ejecutan con indicador de progreso, y recibe la síntesis,
> las salvedades plegables y los seis paneles de verificación con una
> leyenda de escalas para interpretarlos. No hay historial ni
> comparación entre imágenes: es una herramienta de análisis de una foto
> a la vez.

## Principios de diseño y restricciones reales

> Este trabajo no partió de unos requisitos previos, sino de un
> conjunto de principios y restricciones que se declaran de forma
> explícita porque sostienen decisiones de diseño que de otro modo
> parecían arbitrarias.
>
> **Verificabilidad.** Es el principio que gobierna toda la arquitectura,
> con su enunciado completo en el apartado 4.4. Se traduce en un
> contrato uniforme: una clase MetricaConfianza envuelve cada métrica
> con su valor, confianza binaria y fuente_confianza. Esto obliga a cada
> especialista a declarar la aplicabilidad de cada medición que emite, y
> al crítico a anclar toda cifra de su síntesis a un campo existente en
> un informe. Es también el principio
> que impone la separación entre informe (autoría del motor de cómputo,
> recomputable) y diagnóstico (autoría del modelo de lenguaje,
> verbalización), que atraviesa los cuatro especialistas y permite medir
> por separado la fidelidad de la transcripción y de la métrica-texto.
>
> **Reproducibilidad.** Un sistema que basa su credibilidad en la
> verificabilidad no puede permitirse que la misma imagen produzca
> resultados distintos entre dos ejecuciones. Se detectó que cv2.grabCut
> no es determinista por defecto (su inicialización interna consume el
> generador de números aleatorios global de OpenCV, sin exponer semilla
> propia), y lo mismo ocurre con *k-means*. Los apartados 6.4.5 y 6.5.5
> detallan cómo se resolvió cada caso.
>
> **Explicabilidad.** No basta con que el sistema produzca una salida
> razonable: tiene que ser posible rastrear de dónde sale cada
> afirmación. Se aplica Grad-CAM sobre el clasificador de contexto, y
> los seis paneles se dibujan sobre la imagen mostrando lo que cada
> componente ha medido, de modo que una afirmación como
> `[ratio_nitidez.valor = -0.9124]` tenga un respaldo visual que la haga
> legible y no sea solo un número en JSON.
>
> **Coste y latencia.** La arquitectura es híbrida: toda la visión por
> computador corre en local sobre GPU propia y el modelo de lenguaje se
> ejecuta en la nube, a través de la API de Google Gemini. Ese reparto
> mantiene fuera de la nube el cómputo que sostiene la verificabilidad
> del sistema y acota el gasto de API a las seis llamadas del crew ---una
> por tarea---, más la de la lectura visual cuando esa rama está activa:
> hasta 7 llamadas por análisis, desglosadas en el apartado 9.3. Los
> especialistas incorporan gemini-2.5-flash, ya que solo tienen que
> diagnosticar a partir de las métricas, mientras que el agente
> crítico se fija en gemini-2.5-pro pese a su mayor coste y latencia,
> por un motivo que no era previsible y que se explica en el apartado
> 4.7.2.
>
> Sobre estos principios operan tres restricciones reales del
> proyecto, no elegidas sino dadas:
>
> **GPU única.** El desarrollo se realizó sobre una GPU personal. Esto
> explica la elección de yolo11m en lugar de variantes mayores y de un
> ResNet18 con ajuste fino ligero en dos etapas en lugar de un
> entrenamiento desde cero.
>
> **VRAM.** Esta restricción desaconseja desplegar el sistema en la nube,
> porque introduce otra capa de complejidad que no es objetivo del
> proyecto.
>
> **API de pago.** El uso de Gemini tiene coste y cuota por clave, lo que
> descarta cualquier despliegue público sin control de acceso: una
> instancia abierta expondría la clave API a un consumo deliberado.
> Refuerza así la decisión de mantener el sistema en local.

## Metodología y desarrollo de la calibración

> Los umbrales que deciden si una métrica es citable no se fijaron por
> intuición ni se copiaron de la literatura sin comprobación propia: los
> diez del nivel 2 y los cierres del
> crítico se calibraron siguiendo el mismo procedimiento en cuatro
> fases, aplicado de forma independiente en cada condición de
> aplicabilidad.

### Ciclo contrato provisional -\> sonda de calibración -\> revisión visual -\> contrato cerrado

> **Contrato provisional.** Antes de programar se anota qué mide cada
> métrica, su dominio y su cota teórica, y los modos de fallo que se
> pueden anticipar sin haber visto todavía ningún dato. Por ejemplo,
> cv2.grabCut no tiene forma de detectar por sí solo que se le ha dado
> un rectángulo de inicialización vacío, y el filtro de
> saturación/valor de la armonía cromática puede dejar pasar ruido en
> una imagen en blanco y negro. Los umbrales se dejan sin fijar en esta
> fase.
>
> **Sonda de calibración.** Se construye un script auxiliar que sirve
> como base para integrar las distintas técnicas de visión artificial y
> probarlas sobre imágenes de prueba; esos archivos no forman parte del
> sistema. Con la evidencia que producen se decide el comportamiento de
> cada regla compositiva antes de implementarla en los agentes como
> herramienta.
>
> **Revisión visual.** Se miran las imágenes para comprobar si realmente el
> algoritmo de visión había hecho bien su trabajo o si fallaba. Por
> ejemplo, la condición de aplicabilidad de la convergencia perspectiva
> del agente de líneas y dirección abría en falso ante un muro de
> textura, una masa de vegetación o una escena de fauna; por eso pasó de
> dos condiciones a tres. La del horizonte, en el mismo agente, pasó de
> una condición a tres al comprobar a ojo que se detectaba un horizonte
> en el patrón de un muro, en el borde de una colina y en una línea de
> fuga de una escena sin horizonte.
>
> **Contrato cerrado.** Con la evidencia de la sonda de calibración y la
> verificación visual se fijan los umbrales y su justificación, y esos
> valores se trasladan al código. Si más adelante un umbral necesita
> revisarse, hay que repetir la revisión visual antes de recalibrarlo.

### Criterio de banda vacía

> Un umbral de aplicabilidad se considera defendible cuando, al ordenar los
> valores medidos sobre las imágenes, cae dentro de un intervalo sin
> ninguna observación, una separación entre el conjunto que debe abrir
> la condición de aplicabilidad y otra que debe cerrarla. Es el criterio con el que se
> fijó, entre otros, el corte de apertura del haz de convergencia
> perspectiva (35º entre un grupo de casos de paralelismo que no supera
> los 30º y un grupo de perspectiva real que no baja de 42º, sobre las
> 25 imágenes con las que se calibró).

# Diseño de la arquitectura

Con el encargo y el método de calibración ya fijados, este capítulo describe la
arquitectura que los materializa, todavía sin entrar en cada componente: los
tres niveles y el flujo de datos entre ellos, la separación en cuatro
dimensiones compositivas, los contratos que cruzan de un nivel al siguiente, el
contrato de confianza por métrica del que se deriva el resto del diseño, la
doble autoría de informe y diagnóstico, el vector de pesos contextuales W y su
aislamiento, las tecnologías elegidas y la trazabilidad visual como principio.
Los capítulos 5, 6 y 7 desarrollan después un nivel cada uno.

## Visión general: tres niveles, seis agentes y flujo de datos

> El sistema implementa una arquitectura multiagente jerárquica de tres
> niveles, construida sobre CrewAI, en la que cada nivel tiene una
> responsabilidad distinta y no repetida por ningún otro. La jerarquía no
> es de mando en el tiempo de ejecución (se justifica en el apartado 2.6); es una
> descomposición de responsabilidad.
>
> **Nivel 1 -- Agente orquestador (contexto y percepción compartida).** Un único
> agente (`orchestrator_agent`) ejecuta dos herramientas deterministas
> antes de que ningún LLM entre a interpretar nada:
> `ContextClassifierTool` (ResNet18 con ajuste fino sobre EVA, cinco
> clases + rechazo por umbral, con Grad-CAM) y `SharedPerceptionTool`
> (YOLO para el sujeto principal + mapa de saliencia por residuo
> espectral, calculado siempre). Su salida estructurada
> (SalidaOrquestador, validada con `output_pydantic`) fija el terreno
> común sobre el que trabaja el resto del sistema: la etiqueta de
> contexto y la percepción compartida (`bbox`, sujeto_discreto ---el indicador
> booleano que dice si hay un objeto localizable sobre el que anclar una
> medición, definido en el apartado 5.2.3---, mapa de
> saliencia). Deliberadamente no incluye el vector W ---su aislamiento se
> explica en el apartado 4.6---, de modo que es estructuralmente imposible
> llegar a W desde el nivel 2 a través del grafo de datos.
>
> **Nivel 2 -- Cuatro especialistas compositivos.**
> composición_espacial_agent, líneas_direccion_agent, luz_tono_agent y
> espacio_aislamiento_agent miden, cada uno con su propia herramienta de
> visión por computador clásica (NumPy/OpenCV/scikit-learn), una
> dimensión compositiva independiente de las otras tres: encuadre en el
> plano 2D, líneas y perspectiva, luz y color, y aislamiento
> figura-fondo. Todos comparten el mismo contrato de salida, un informe
> que produce el motor y un diagnóstico en prosa que redacta el LLM
> citando ese informe (ver el apartado 4.5), y los cuatro se ejecutan
> siempre, sin que ningún componente decida si «aplican» a la imagen.
> Lo que varía entre imágenes no es si el especialista corre, sino si
> sus métricas resultan citables, y eso lo deciden las condiciones de
> aplicabilidad dentro de cada herramienta (apartado 4.4), nunca un LLM.
> Dos de los cuatro (líneas y dirección; luz y tono) son puramente globales
> y no consumen la percepción compartida; los otros dos
> (composición espacial; espacio y aislamiento del sujeto) sí la usan para anclar su
> medición al sujeto.
>
> **Nivel 3 -- Agente crítico.**
>
> critico_agent es el único agente que conoce las cuatro dimensiones a
> la vez y el único que recibe W (vía `PrioridadTool`, que lo consulta
> por su cuenta sobre parámetros/pesos_contextuales.json). Su salida
> CriticaCompositiva separa igualmente una parte determinista y
> recomputable (el arbitraje, calculado sin LLM) de la síntesis en prosa,
> que se obtiene de los cuatro diagnósticos, resuelve las tensiones entre
> dimensiones y, de forma opcional, incorpora una tercera fuente de
> información: una lectura visual ciega a las métricas (apartado 7.4).
>
> **Flujo de datos.** Los tres niveles se corresponden uno a uno con seis
> agentes y seis tareas de CrewAI, ejecutadas bajo Process.sequential,
> no Process.hierarchical (su justificación está en el apartado 2.6). El acoplamiento
> entre tareas no es el encadenado por defecto de CrewAI, sino un
> parámetro `context` por tarea: los dos especialistas globales reciben `context=[]`,
> y solo la tarea del crítico recibe las cinco anteriores completas.
> Esto es lo que impide, por construcción y no por una
> instrucción, que una dimensión silencie o acapare a otra o que W llegue
> antes de tiempo. Cada tarea persiste su salida tipada en fichero JSON
> propio dentro de outputs/ejecucion/{stem}/, que funciona
> simultáneamente como el producto final del sistema y como el canal de
> estado entre niveles. La Tabla 4.1 resume, tarea por tarea, el nivel,
> el contexto de entrada y la salida tipada.

**Tabla 4.1.** Agentes y tareas del sistema, con su nivel, el contexto
que reciben y la salida tipada que producen.

  ---------------------------------------------------------------------------------------
  Agente                       Nivel   Entrada (contexto) Salida tipada
  ---------------------------- ------- ------------------ -------------------------------
  orchestrator_agent           1       {ruta_imagen}      SalidaOrquestador

  composicion_espacial_agent   2       `[compose_task]`   DiagnosticoComposicionEspacial

  líneas_direccion_agent       2       `[]`               DiagnosticoLineasDireccion

  luz_tono_agent               2       `[]`               DiagnosticoLuzTono

  espacio_aislamiento_agent    2       `[compose_task]`   DiagnosticoEspacioAislamiento

  critico_agent                3       Las 5 tareas       CriticaCompositiva
                                       anteriores         
  ---------------------------------------------------------------------------------------

> [PENDIENTE-AUTOR: falta la Figura 4.1 con el diagrama de la arquitectura de tres
> niveles y el flujo de las seis tareas de la Tabla 4.1 --- los tres niveles como
> bloques, las seis tareas como flechas etiquetadas con su `context` real ( `[]` para
> líneas y dirección y luz y tono, `[compose_task]` para composición espacial y
> espacio y aislamiento del sujeto, las cinco anteriores para el crítico) y el punto
> exacto por el que W no pasa, para que la figura muestre el aislamiento arquitectónico
> del apartado 4.6 en vez de solo describirlo en prosa.]

## Separación funcional: cuatro dimensiones compositivas

> En la composición fotográfica existen varias técnicas que organizan
> los elementos de la imagen, y algunas pueden agruparse por
> dimensiones. Esa agrupación asegura que dos dimensiones no se hagan la
> misma pregunta y que se solapen lo menos posible. En la literatura de
> Freeman y Präkel se separan
> estas dimensiones compositivas por capítulos, además de que organizan el
> juicio alrededor de preguntas que no comparten evidencia entre sí:

- ¿Dónde está el sujeto en el plano del encuadre?

- ¿Qué estructura geométrica organiza la mirada?

- ¿Cómo se relacionan la figura y el fondo?

- ¿Cómo se distribuyen la luz y el color?

> También abarcan otras preguntas que no son objetivo del sistema
> actualmente. Esas cuatro son la base de los cuatro especialistas y del
> vector de pesos contextuales W
> (parámetros/pesos_contextuales.json), que asigna más importancia a
> unas dimensiones que a otras, y no a los agentes. Esta distinción
> es relevante porque un peso sobre un agente sería un juicio sobre un
> componente software, mientras que un peso sobre una dimensión
> compositiva es un juicio fotográfico, que es lo que W tiene que ser.

## Contratos de datos entre niveles y aislamiento del contexto de cada tarea

> **Contratos validados.** Los contratos son esquemas Pydantic validados, no una convención de
> redacción.
>
> Cada tarea de CrewAI declara `output_pydantic = <Modelo>`, lo que
> significa que la salida de cada agente no es simplemente texto que se
> parece a JSON: CrewAI fuerza la respuesta del LLM contra el esquema
> declarado (schemas/orquestador.py, schemas/especialistas.py,
> schemas/critico.py) y la tarea falla si no encaja. Esto convierte cada
> contrato en una propiedad mecánica y no en una promesa del backstory
> que el modelo podría incumplir. Lo que ese mecanismo no garantiza es el contenido ---una
> fila de pesos inventada tiene la misma forma que la correcta, como
> muestra el apartado 7.2.5---, y de eso se ocupan las condiciones de
> aplicabilidad y el arbitraje determinista.

### El bloque compartido: MetricaConfianza

> El contrato común a todos los especialistas es un envoltorio genérico:
>
> `MetricaConfianza[T] = {valor: T, valor_norm: float | None, confianza: float, fuente_confianza: str | None}`
>
> Cada métrica de cada especialista tiene esta forma, y eso permite que
> `PrioridadTool`, en el nivel 3, la detecte por su estructura.

### El contrato del nivel 1 al nivel 2: SalidaOrquestador

> `SalidaOrquestador = { etiqueta_contexto: str, percepcion_compartida: PercepcionCompartida }`
>
> `PercepcionCompartida` son diez campos fijos (`bbox`, `clase`,
> `confianza_yolo`, `fuente`, `sujeto_discreto`, `blob_area_ratio`,
> `mapa_saliencia_path`, `centroide_saliencia`, `deteccion_yolo_path`,
> `percepcion_bbox_path`).

### El contrato del nivel 2 al nivel 3

> Aquí la información del nivel 2 al nivel 3 fluye por dos canales
> distintos:

- El canal de contexto de CrewAI (critica_task con
  `context=[compose_task, composicion_espacial_task, lineas_direccion_task, luz_tono_task, espacio_aislamiento_task]`): es
  el que entrega al LLM del crítico el contenido de los diagnósticos de
  los especialistas, con él llegan los valores de las métricas,
  que es lo que el crítico necesita para poder citarlas en su síntesis.

- El canal de la ejecución en disco
  (outputs/ejecucion/{stem}/salida\_\*.json, leído directamente por
  PrioridadTool sin pasar por ningún LLM): es el que produce el
  arbitraje, qué dimensiones son citables, en qué orden por W, con qué
  cobertura. El segundo canal es el que decide la estructura del discurso,
  y el primero el que aporta el contenido con el que se rellena esa
  estructura.

### El aislamiento de contexto entre tareas: context=[]

> Process.sequential es el flujo que se ha decidido en el sistema, pero
> por defecto CrewAI encadena todas las salidas de las tareas anteriores
> en el parámetro `context` de cada nueva tarea. Eso dejaría, por
> ejemplo, la cuarta tarea del equipo (luz_tono_task) con los informes
> del orquestador y de los especialistas que la preceden, y contaminaría
> con evidencia ajena una dimensión que el apartado 4.2 quiere mantener
> separada. El sistema evita este comportamiento declarando el parámetro
> `context` de cada tarea de forma explícita.

### Aislamiento entre ejecuciones

> Cada análisis escribe en outputs/ejecucion/{stem}, un directorio
> propio por imagen. Dos análisis de imágenes distintas no comparten
> ubicación.
>
> Los contratos de este sistema no solo especifican la forma de lo que
> cruza entre niveles, sino que incluyen mecanismos para impedir que
> cruce la información que no debe.

## Contrato de confianza por métrica: condiciones de aplicabilidad y prohibición de falsa precisión

> Todo el diseño del nivel 2, y parte de la memoria dedicada a él en el
> capítulo 6, son la consecuencia de aplicar consistentemente dos
> prohibiciones fijadas antes de escribir el primer especialista:

- Ningún especialista fabrica una métrica de falsa precisión. Un algoritmo
  como cv2.grabCut o el consenso RANSAC de un punto de fuga siempre
  devuelve un resultado, tenga o no sentido la pregunta sobre esa imagen
  (un retrato no tiene punto de fuga, pero RANSAC encontrará uno si se
  insiste). Si el sistema publicara ese número sin más, el crítico lo
  citaría como un hecho verificable cuando en realidad es ruido con
  apariencia de medición.

- Ningún especialista silencia una métrica: La respuesta a «esto no
  aplica» no puede ser omitir el campo, porque un campo ausente es
  indistinguible de un fallo de transcripción, y dejaría al crítico sin
  ninguna traza de que la pregunta se llegó a formular.

> La resolución de esta tensión es el contrato de confianza por métrica:
> cada magnitud publicada se envuelve en `MetricaConfianza[T]` y una condición de aplicabilidad
> determinista, resuelta por el mismo motor que calcula el valor,
> decide si confianza vale 1 o 0. El valor se publica siempre: lo que
> cambia es si es citable o no.

> Una condición de aplicabilidad no es un umbral aislado: es el conjunto de los modos de fallo
> de todas las señales de las que esa métrica concreta depende. Eso
> explica por qué el número de condiciones varía de una métrica a otra:
> cada una hereda los de sus propias fuentes de evidencia, no una lista
> genérica de precauciones.
>
> Dos de los criterios de la Tabla 4.2 se apoyan en `fuente`, el campo de
> la percepción compartida que registra de dónde salió el cuadro delimitador del
> sujeto ---de una detección de YOLO o de la saliencia---, y que el apartado
> 5.2.3 desarrolla.

**Tabla 4.2.** Condición de aplicabilidad de cada métrica citable, por
especialista.

  ---------------------------------------------------------------------------
  Especialista           Métrica                  Criterios de aplicabilidad
  ---------------------- ------------------------ ---------------------------
  Composición            d_tercios, d_centro,     `fuente == "yolo"`
  espacial               patron_dominante

  Composición            d_equilibrio             Ninguna
  espacial

  Líneas y               angulo_horizonte         Existencia de candidato +
  dirección                                       longitud horizonte \>=
                                                  0.13 + ángulo \<= 10 (3
                                                  condiciones)

  Líneas y               score_convergencia       n_lineas_fuga \>= 2 + score
  dirección                                       \>
                                                  score_convergencia_nulo +
                                                  apertura_haz \>= 35º

  Espacio y              ratio_espacio_negativo   `fuente == "yolo"`
  aislamiento del sujeto

  Espacio y              ratio_nitidez            `fuente == "yolo"` +
  aislamiento del sujeto                          max(var_fig, var_fondo) \>=
                                                  50

  Luz y tono             media_L                  ninguna

  Luz y tono             esquema_cromatico        ratio_pixeles_cromaticos
                                                  \>= 0.10
  ---------------------------------------------------------------------------

## Doble autoría informe recomputable frente a diagnóstico verbalizado

> Si el contrato de confianza decide qué se puede afirmar, la doble autoría
> decide quién lo firma. Este apartado explica el problema que la motivó, qué
> tiene que aportar el diagnóstico para que el paso por el modelo de lenguaje no
> sea un coste sin retorno, y las reglas de redacción que lo gobiernan.
>
> [PENDIENTE-AUTOR: falta una figura con el contrato `informe`/`diagnostico`
> y sus dos autorías --- dos cajas por especialista (`informe`, firmado por el motor
> determinista; `diagnostico`, firmado por el LLM citando ese informe) con la flecha
> de dependencia de una a la otra, repetido y generalizado a los cuatro especialistas
> y al arbitraje/síntesis del crítico --- para que la separación de autoría se vea de
> un vistazo en vez de tener que reconstruirla leyendo los apartados 4.5.1 a 4.5.3.]

### El problema que este diseño resuelve

> La primera versión del Agente 1 no separaba el informe del
> diagnóstico: el LLM del especialista recibía el JSON de su herramienta
> y lo transcribía en prosa al siguiente agente. Eso es algo que ya hace
> `json.dumps()`, así que el LLM de cada especialista no aportaba nada
> que el script no aportara ya, y mantenerlo solo complicaba la
> arquitectura.
>
> **Solución adoptada.** Cada especialista emite dos objetos con autoría distinta:
>
> `DiagnosticoX = {informe: InformeX, diagnostico: str}`

- El informe lo produce el motor (NumPy/OpenCV/scikit-learn dentro de
  la herramienta) y constituye la parte auditable y recomputable: cualquier
  tercero con la misma imagen y el mismo código obtiene el mismo
  informe, campo a campo.

- El diagnóstico lo redacta el LLM del especialista, en 4-6 frases,
  citando ese informe, y convierte cifras en hechos
  compositivos ya interpretados para que el crítico del nivel 3 pueda
  razonar sobre cuatro párrafos y un vector W en lugar de sobre treinta
  números sueltos.

> Esta separación del motor que calcula y el LLM que verbaliza es un
> patrón que se repite en los cuatro especialistas y, una capa más
> abajo, en el propio crítico.

### Qué tiene que aportar el diagnóstico para justificar su existencia

> Si el diagnóstico no hace más que repetir el informe con otras
> palabras, sigue siendo el mismo problema; por ello se exige un
> contrato que aporte cinco cosas que ninguna transcripción automática
> da:

- El diagnóstico aporta un veredicto claro: no solo el número, sino qué
  dice ese número sobre la fotografía.

- También traduce las coordenadas a lenguaje espacial: «tercio inferior
  izquierdo» en vez de (0,31; 0,68).

- Indica la dirección de desplazamiento, deducida del signo, y no solo la
  magnitud (p. ej., hacia qué lado se inclina el horizonte, y no solo
  cuántos grados).

- Relaciona las señales dentro de la misma dimensión: cuando el
  sujeto y el peso visual global coinciden es refuerzo compositivo;
  cuando están en lados opuestos es contrapeso. Esta es la lectura que
  ninguna métrica aislada da por sí sola.

- Explicita las salvedades de fiabilidad: qué no se puede afirmar y por qué,
  leyendo la confianza y fuente_confianza del contrato del apartado 4.4.

### Las reglas de la redacción «plantilla de instrucciones de los agentes» y su justificación

> El description de cada campo diagnóstico y el backstory de cada agente
> imponen un conjunto de reglas que, tomadas juntas, son la traducción
> operativa de la premisa de verificabilidad a nivel de prosa:

- Toda afirmación se cita con notación de punto sobre un escalar
  (`[d_tercios.valor = 0.0648]`), nunca pegando el diccionario entero.
  Lo que no se puede citar, no se afirma.

- El especialista no habla de importancia, prioridad o gravedad. El nivel 2 no
  tiene W (vector de pesos contextuales). El especialista dice qué pasa
  en su dimensión; cuánto pesa lo decide el crítico. Es la misma
  ortogonalidad que separa confianza (fiabilidad de la medida) de la
  importancia contextual en MetricaConfianza.

- El especialista no opina sobre las otras tres dimensiones ni emplea
  adjetivos estéticos (buen equilibrio o métrica acertada).

- Los adverbios de grado se emplean solo si la cifra los respalda con una escala
  documentada. Sin ella el LLM inventa calificativos: fue el primer
  fallo medido en el Agente 1, que llamó «clara adherencia» a un margen
  por debajo de la media observada.

## Vector de pesos contextuales W: definición, justificación bibliográfica y aislamiento arquitectónico

> Las cuatro dimensiones no pesan lo mismo en todos los géneros fotográficos, y
> el sistema lo declara en un solo sitio: el vector de pesos contextuales W.
> Este apartado dice qué es, de dónde salen sus valores y por qué está aislado
> del resto del grafo de agentes.

### Qué es W

> La matriz W reúne los vectores de pesos en una tabla 6x4, recogida en
> la Tabla 4.3: seis contextos fotográficos (los cinco que
> reconoce el clasificador de nivel 1 más la categoría de rechazo
> «otro») por las cuatro dimensiones compositivas del apartado 4.2, con
> cada fila sumando 1. Es una distribución de importancia relativa, no
> una puntuación absoluta.

**Tabla 4.3.** Matriz de pesos contextuales W: importancia relativa de
cada dimensión compositiva por contexto fotográfico.

  -----------------------------------------------------------------------------
  Contexto              Composición   Líneas y    Espacio y              Luz y
                        espacial      dirección   aislamiento del sujeto tono
  --------------------- ------------- ----------- ---------------------- ------
  Animal                0,24          0,14        0,32                   0,30

  Arquitectura/urbana   0,22          0,37        0,17                   0,24

  Retrato/humano        0,22          0,10        0,31                   0,37

  Paisaje natural/rural 0,24          0,19        0,22                   0,35

  Producto/bodegón      0,26          0,11        0,23                   0,40

  Otro (Rechazo)        0,25          0,25        0,25                   0,25
  -----------------------------------------------------------------------------

> La fila de «otro» indica la categoría de las imágenes que no han sido
> detectadas con un contexto de los establecidos con una cierta
> confianza.
>
> [PENDIENTE-AUTOR: falta una figura con el aislamiento de W en el grafo de
> tareas --- las seis tareas con sus aristas de `context` reales (apartado 4.3.4) y W
> dibujado como un nodo que solo conecta con la tarea del crítico a través de
> `VectorPesosTool`, nunca por el grafo de contexto --- para hacer visible de un
> vistazo la propiedad que el apartado 4.1 ya declara en prosa: que es
> estructuralmente imposible llegar a W desde el nivel 2.]

### Por qué W

> La hipótesis de partida procede de la propia literatura fotográfica y
> de la evidencia experimental recogida en el conjunto de datos EVA: la
> importancia relativa de cada dimensión compositiva no es la misma en
> todos los géneros. Freeman y Präkel organizan el juicio compositivo
> alrededor de recursos distintos según el tipo de escena ---las líneas
> convergentes vertebran la fotografía de arquitectura, mientras que en
> producto el sujeto suele estar centrado y son la luz y el color los
> que construyen el volumen y la textura---, y los autores de EVA
> observan lo mismo de forma cuantitativa al segmentar sus anotaciones
> por categoría de contenido (apartado 2.2). Un sistema que ponderase
> las cuatro dimensiones por igual en todas las fotografías estaría
> contradiciendo esa evidencia.
>
> Fijar los valores concretos de esa ponderación plantea otro problema,
> porque no existe una verdad de referencia contra la que validarlos: la
> importancia relativa de una dimensión compositiva no es una magnitud
> medible. El procedimiento seguido tuvo tres pasos, de los cuales solo
> el último fija los números.

**Punto de partida: sondeo de percepción.** Antes de recurrir a la
bibliografía se recogió una primera referencia numérica preguntando
directamente por la importancia de cada dimensión en cada contexto. Los
valores promedio, normalizados para que cada fila sume 1, son los que
recoge la Figura 4.1:

> ![](docs/memoria/media/media/image1.png){width="4.707638888888889in"
> height="2.0172430008748905in"}
>
> **Figura 4.1.** Resultado del sondeo de percepción: importancia
> promedio normalizada de cada dimensión compositiva, por contexto
> fotográfico. No fija los pesos de W; solo comprueba que la hipótesis
> de partida no es contraintuitiva (ver el texto siguiente).

> Su función es acotada: el sondeo no fija los pesos, sino que comprueba
> que la hipótesis no es contraintuitiva. La dimensión que la matriz final señala como
> dominante ocupa también el primer puesto del sondeo en cuatro de los
> cinco contextos ---líneas y dirección en arquitectura, luz y tono en
> paisaje, y, empatadas con otra dimensión, luz y tono en retrato y
> espacio y aislamiento del sujeto en animal--- y discrepa en producto, donde
> el sondeo sitúa la luz y el tono en último lugar mientras que la
> matriz final la sitúa en primero, siguiendo a la literatura de
> fotografía de producto. Hay
> además una limitación del instrumento que impide leer sus magnitudes
> como pesos: al puntuarse cada dimensión por separado y normalizar
> después, las diferencias se comprimen, y todos los valores quedan
> entre 0,22 y 0,29, es decir, muy cerca del reparto uniforme. El sondeo
> informa del orden; no de la distancia entre dimensiones.

**Contraste con modelos de lenguaje.** Como segundo punto de referencia
se pidió la misma matriz a tres modelos de lenguaje generales, con una
única instrucción idéntica para los tres, en conversaciones nuevas y sin
contexto previo. La instrucción completa se recoge en el anexo
correspondiente. Las Figuras 4.2 a 4.4 reproducen, reencuadradas, las
tres respuestas.

> Chat GPT 5.6 -- Sol
>
> ![](docs/memoria/media/media/image2.png){width="4.322948381452319in"
> height="1.54167760279965in"}
>
> **Figura 4.2.** Matriz de pesos propuesta por Chat GPT 5.6 «Sol» ante
> la instrucción del anexo correspondiente. Se conserva reencuadrada
> como contraste del orden de magnitud, no como evidencia de los pesos
> finales (ver el texto siguiente).
>
> Gemini 3.1 -- Pro
>
> ![](docs/memoria/media/media/image3.png){width="4.989619422572178in"
> height="3.8229451006124235in"}
>
> **Figura 4.3.** Matriz de pesos propuesta por Gemini 3.1 Pro ante la
> misma instrucción. Se conserva reencuadrada como contraste del orden
> de magnitud, no como evidencia de los pesos finales.
>
> Claude Opus 5 Max
>
> ![](docs/memoria/media/media/image4.png){width="3.4791918197725282in"
> height="2.2291830708661418in"}
>
> **Figura 4.4.** Matriz de pesos propuesta por Claude Opus 5 Max ante
> la misma instrucción. Se conserva reencuadrada como contraste del
> orden de magnitud, no como evidencia de los pesos finales.

> Estas tablas tampoco son la fuente de los valores finales, por dos
> motivos. El primero es de fondo: estos modelos
> se han entrenado sobre la misma literatura fotográfica que aquí se
> cita, de modo que su coincidencia con Freeman o Präkel no constituye
> una confirmación independiente, sino la misma fuente por otra vía. El
> segundo es aritmético: la media de las tres tablas no reproduce la
> matriz final. En el contexto de retrato, por ejemplo, ese promedio
> situaría el espacio y aislamiento del sujeto como dimensión dominante, mientras
> que la matriz final da ese papel a la luz y el tono, apoyándose en la
> correlación que EVA documenta para la categoría de personas. Se
> recogen como contraste del orden de magnitud y de la ordenación entre
> dimensiones, no como evidencia.

**Fijación de los valores: razonamiento bibliográfico fila a fila.** Los
pesos definitivos se fijaron uno a uno tomando como referencia la teoría
fotográfica clásica y las correlaciones de EVA, junto con los dos pasos
anteriores. Cada fuente aporta algo distinto: la bibliografía
sostiene qué dimensión domina en cada género y en qué orden vienen las
demás, que es una afirmación que Freeman, Präkel y EVA sí respaldan; la
magnitud concreta de cada peso es una asignación del autor, coherente con
ese orden pero no deducible de ninguna de las fuentes. Es el mismo límite
que ya se declaró para el sondeo de percepción, y la razón por la que la
matriz completa se presenta como hipótesis de diseño y no como un
parámetro estimado.

- **Animal (0,24 · 0,14 · 0,32 · 0,30).** La fotografía de fauna prioriza
  el aislamiento del sujeto respecto al entorno (espacio y aislamiento del sujeto:
  0,32) y una iluminación que revele texturas y detalles del animal (luz
  y tono: 0,30). La composición espacial es relevante pero secundaria
  (0,24), y la estructura lineal rara vez organiza la mirada en escenas
  naturales con animales (0,14). Freeman destaca que en
  fotografía de naturaleza el aislamiento del sujeto y la luz son los
  factores dominantes.

- **Arquitectura y escenas urbanas (0,22 · 0,37 · 0,17 · 0,24).** Las
  líneas de fuga, la simetría de las fachadas y la dirección visual son
  el recurso compositivo central (líneas y dirección: 0,37). La
  composición espacial y la luz son relevantes (0,22 y 0,24), y el
  aislamiento del sujeto es el factor menos determinante (0,17), porque
  en estas escenas el sujeto suele ser la estructura completa y no un
  objeto recortable. Präkel identifica las líneas convergentes como el
  elemento compositivo principal en arquitectura.

- **Retrato y figura humana (0,22 · 0,10 · 0,31 · 0,37).** La
  iluminación es el factor más importante (luz y tono: 0,37), seguida
  del aislamiento del sujeto mediante desenfoque de fondo o contraste de
  nitidez (espacio y aislamiento del sujeto: 0,31). La composición espacial es
  moderada (0,22) y la estructura lineal es poco relevante (0,10). Es la
  fila donde la evidencia de EVA es más directa: la correlación entre
  luz y color y la puntuación estética alcanza su máximo en la categoría
  de personas.

- **Paisaje natural y escenas rurales (0,24 · 0,19 · 0,22 · 0,35).** La
  distribución es más equilibrada, con predominio de la luz (0,35) y de
  la composición espacial (0,24). Las líneas ---horizonte, caminos---
  tienen peso moderado (0,19) y el aislamiento del sujeto es menos
  crítico cuando la escena completa es el sujeto (0,22). Freeman señala
  que en paisaje la regla de los tercios y la luz de las horas extremas
  del día son los recursos más citados.

- **Producto y bodegón (0,26 · 0,11 · 0,23 · 0,40).** La iluminación es
  dominante (0,40), porque define la textura, el volumen y el atractivo
  del producto. La composición es relevante (0,26) y el aislamiento del
  sujeto, moderado (0,23). La estructura lineal es prácticamente
  irrelevante sobre fondos controlados (0,11). Es la fila en la que el
  sondeo inicial discrepaba, y se resolvió a favor de la bibliografía.

- **Otro, categoría de rechazo (0,25 · 0,25 · 0,25 · 0,25).** Pesos
  uniformes que se aplican cuando el clasificador de contexto rechaza la
  imagen por baja confianza. Es una decisión conservadora: sin un
  contexto fiable no hay base para privilegiar ninguna dimensión, y el
  reparto uniforme es el único que no introduce un sesgo que el sistema
  no pueda justificar.

> El alcance de todo esto es el siguiente: W es una hipótesis de diseño,
> no un parámetro aprendido. No se ha estimado a
> partir de datos etiquetados ni se ha optimizado contra ninguna
> métrica, y el trabajo no afirma que estos pesos sean los óptimos, sino
> que son explícitos, justificados y auditables. Cualquiera puede
> abrir el fichero de parámetros, discrepar de una fila y comprobar qué
> cambia en la crítica, porque W no se transcribe en ninguna instrucción
> ni viaja por el grafo de agentes. Esta limitación queda recogida como
> tal en el apartado 9.5.

## Selección de tecnologías y modelos

### La decisión estructural: visión en GPU local, síntesis en la nube

> El sistema reparte su cómputo en dos dominios de ejecución distintos:
> la visión por computador (clasificador ResNet, YOLO, saliencia y los
> motores del nivel 2) corre en local, sobre GPU, con librerías
> deterministas y auditables (NumPy, OpenCV, scikit-learn, PyTorch),
> mientras que los informes de los agentes y la síntesis del crítico se
> ejecutan en la nube de Google Gemini.
>
> Además de esas llamadas, el equipo se construye con la traza de ejecución de
> CrewAI activada (`tracing = True`), que envía los eventos de cada tarea a
> la plataforma del propio *framework*; es instrumentación de desarrollo,
> no interviene en el análisis ni en sus resultados, y se desactiva con
> esa misma constante.

### Tecnologías y modelos

- El entorno emplea Python 3.12.10 y el gestor uv; CrewAI 1.15.13 (Google-genai);
  PyTorch con CUDA 12.4; ultralytics (YOLO); opencv-contrib-python
  por cv2.saliency (el mapa de residuo espectral de la percepción
  compartida); scikit-learn para *k-means* circular del agente de luz
  y tono.

- El clasificador de contexto utiliza ResNet18 con ajuste fino sobre EVA,
  cinco clases más rechazo por umbral a «otro», con Grad-CAM para
  explicabilidad. Es el único componente entrenado del sistema.

- La detección del sujeto utiliza yolo11m.pt, decidido tras evaluar varios
  pesos de YOLO y descartar los que ofrecen el mismo resultado con
  mayor latencia.

- El crítico utiliza gemini-2.5-pro, y no el mismo gemini-2.5-flash que usan
  los especialistas. El crítico es un agente que recibe todos los
  informes a la vez y dos herramientas (arbitraje determinista +
  lectura visual), así que su plantilla de instrucciones triplica la de cualquier
  especialista. Con ese tamaño, gemini-2.5-flash devuelve una
  respuesta vacía, mientras que gemini-2.5-pro devuelve la respuesta
  esperada.

## Trazabilidad y verificación visual como principio de diseño

### La trazabilidad no es un añadido de la interfaz: está en cada agente desde el diseño

> La crítica final cita métricas verificables, pero el usuario necesita comprobar a simple vista que la
> medición tiene sentido sobre la fotografía real, sin leer el código.
> Para ello todo agente que hace una afirmación espacial o visual
> produce, además de su informe numérico, una imagen de verificación que
> dibuja exactamente lo que midió sobre la propia fotografía. No es una
> funcionalidad de la interfaz añadida al final: cada especialista
> implementa su panel de verificación como parte de su propio contrato
> de salida, desde el primero que se construyó.
>
> **Inventario de paneles.** La Tabla 4.4 recoge qué prueba cada uno.

**Tabla 4.4.** Inventario de paneles de verificación visual: quién los
genera y qué prueba cada uno.

  --------------------------------------------------------------------------------
  Panel                    Quién lo genera   Qué muestra
  ------------------------ ----------------- -------------------------------------
  Grad-CAM                 nivel 1           Qué regiones de
                           (clasificador)    la imagen pesaron
                                             en la
                                             clasificación de
                                             contexto

  Mapa de saliencia        nivel 1           El mapa de
                           (percepción       residuo espectral
                           compartida)       crudo

  Detecciones YOLO         nivel 1           Todas las
                           (percepción       detecciones,
                           compartida)       incluso por
                                             debajo del umbral
                                             de confianza

  Cuadro delimitador final nivel 1           El `bbox` que utilizan
                           (percepción       efectivamente los
                           compartida)       especialistas,
                                             coloreado según
                                             su origen (YOLO,
                                             saliencia, sin
                                             sujeto)

  Composición              Agente 1          Rejilla de
  espacial                                   tercios, punto de
                                             anclaje, punto
                                             ganador, líneas
                                             de d_tercios,
                                             d_centro y
                                             d_equilibrio.

  Líneas y                 Agente 2          Posibles líneas de
  dirección                                  fuga, líneas de
                                             punto de fuga,
                                             segmento de
                                             horizonte,
                                             veredicto de las
                                             condiciones de aplicabilidad.

  Espacio y                Agente 3          Contorno de la
  aislamiento del sujeto                     máscara, las dos
                                             varianzas del
                                             Laplaciano,
                                             veredicto de las
                                             dos condiciones de aplicabilidad

  Luz y tono               Agente 4          Franja de matices
                                             dominantes,
                                             métricas de
                                             exposición,
                                             veredicto de la
                                             condición de aplicabilidad cromática.
  --------------------------------------------------------------------------------

> Todos comparten una capa de dibujo común (CapaVerificacion), así como
> una misma convención de color, para que cada color signifique siempre
> lo mismo en los seis paneles que la usan.

### Artefactos mostrados en la interfaz

> La trazabilidad se genera entera y se entrega seleccionada. De los
> ocho artefactos que el sistema escribe, la interfaz muestra solo los
> que verifican una medición sobre la fotografía, acompañados de una
> leyenda que explica en qué escala vive cada métrica citada; volcar
> además todos los datos que producen los especialistas no haría al
> usuario más capaz de comprobar nada. Qué se muestra, qué se deja fuera
> y con qué criterio se detalla en los apartados 7.6.4 a 7.6.6.

# Nivel 1 -- Agente orquestador: contexto y percepción compartida

Los tres capítulos siguientes desarrollan un nivel cada uno, en el orden en que
se ejecutan. Este describe el nivel 1, cuyas dos herramientas ---el clasificador
de contexto y la percepción compartida--- son deterministas y corren antes de
que ningún modelo de lenguaje entre a interpretar nada. Termina con la salida
tipada que el nivel 1 entrega al nivel 2 y con la evidencia visual que deja por
el camino.

## Clasificador de contexto

> El clasificador de contexto fotográfico es la primera herramienta que
> ejecuta el agente orquestador (nivel 1) del sistema multiagente. Dada
> una imagen de entrada, su función es asignarle una etiqueta de
> contexto compositivo entre cinco (arquitectura, paisaje,
> retrato/humano, producto y animal), o bien rechazar la clasificación y
> devolver la etiqueta «otro» cuando no reconoce ninguno de ellos o la
> confianza es insuficiente. Esta etiqueta de
> contexto indica qué fila de la matriz W se
> transmite al agente crítico (nivel 3) para ponderar las dimensiones
> compositivas evaluadas por los cuatro especialistas (nivel 2).
> Por tratarse de una decisión que condiciona el comportamiento de todo
> el sistema, su diseño se rigió por dos requisitos no negociables
> derivados del planteamiento del anteproyecto, que parte precisamente
> de que los modelos actuales «actúan como cajas negras»: (1) el
> clasificador debe ser auditable, por lo que se descartó el uso de un
> LLM como clasificador de contexto, y (2) toda predicción de clase debe
> ir acompañada de un mapa de explicabilidad que permita verificar
> visualmente en qué regiones de la imagen se apoyó la decisión. Que ese
> mapa sea un Grad-CAM es ya una decisión de este trabajo, y el rechazo
> a «otro» es su excepción: el apartado 5.1.7 explica por qué no produce
> mapa.

### Selección de conjunto de datos y justificación

> Para entrenar el clasificador se empleó el conjunto de datos EVA (Explainable
> Visual Aesthetics), publicado por Kang, Valenzise y Dufaux [6] en
> el taller ATQAM/MAST'20 de ACM Multimedia. EVA reúne 5101 fotografías
> anotadas con puntuaciones estéticas humanas y, para este subproyecto,
> con seis categorías de contenido semántico:
> animals, architectures and city scenes, human, natural and rural
> scenes, still life y other.
>
> La procedencia de esas categorías determina qué se le puede pedir a la
> etiqueta. Las cinco primeras están
> inspiradas en los tipos de contenido de Liu y Wang [24], como ya se
> señaló en el apartado 2.2. La asignación de cada imagen a su categoría
> es un procedimiento híbrido en dos fases: primero una categorización
> automática, en la que se detectan los objetos de la fotografía con
> YOLOv3 y sus 80 clases preentrenadas se agrupan manualmente en las
> cinco categorías, quedando cada imagen asignada a aquella cuyos
> objetos cubren más de la mitad de la superficie del encuadre; y
> después una revisión humana, en la que tres personas comprobaron la
> validez de la clasificación en cada categoría. De ahí que
> `other` no sea una clase semántica más, sino el residuo de ese criterio
> de área: recoge las imágenes en las que ningún grupo de objetos ocupa
> una proporción mayoritaria del encuadre o en las que compiten varios
> objetos significativos sin un sujeto dominante claro.
>
> El fichero de metadatos publicado junto al conjunto de datos
> (`image_content_category.csv`) codifica estas seis categorías como un
> entero (columna `sort`) sin documentar explícitamente la correspondencia
> código → nombre. Esta se estableció inspeccionando muestras reales de
> imágenes para cada código, y se verificó además que las 5101 filas del
> CSV coinciden exactamente con las 5101 imágenes presentes en disco.

### Mapeo de categorías y partición del conjunto de datos

> Las categorías de EVA están nombradas en inglés, así que el sistema
> las renombra al castellano. La Tabla 5.1 recoge, código a código, el
> nombre de la derecha, que es el que aparece
> literalmente en `class_to_idx.json` y el que viaja como
> `etiqueta_contexto` por todo el resto del sistema.

**Tabla 5.1.** Correspondencia entre las seis categorías originales de
EVA, la clase que usa el sistema y el número de imágenes de cada una.

+----------+--------------+-----------------------+-----------+
| Código   | Categoría    | Clase del sistema     | Nº        |
|          | EVA          |                       | Imágenes  |
+==========+==============+=======================+===========+
| 1        | animals      | `animal`              | 906       |
+----------+--------------+-----------------------+-----------+
| 2        | Architecture | `arquitectura`        | 948       |
|          | and city     |                       |           |
|          | scenes       |                       |           |
+----------+--------------+-----------------------+-----------+
| 3        | human        | `retrato-humano`      | 1034      |
+----------+--------------+-----------------------+-----------+
| 4        | Natural and  | `paisaje`             | 1012      |
|          | rural scenes |                       |           |
+----------+--------------+-----------------------+-----------+
| 5        | Still life   | `producto-still_life` | 928       |
+----------+--------------+-----------------------+-----------+
| 6        | other        | ninguna: no se        | 273       |
|          |              | entrena               |           |
+----------+--------------+-----------------------+-----------+

> `other` de EVA y `otro` del sistema no son lo mismo, aunque el
> nombre invite a confundirlos, y la distinción importa porque son las
> dos puntas de la misma decisión de diseño. `other` es una categoría de
> anotación del conjunto de datos: describe imágenes que existen y que EVA
> etiquetó así por su criterio de área. `otro` es una decisión de
> rechazo del sistema en tiempo de inferencia: no es una salida del
> modelo, sino lo que se emite cuando ninguna de las cinco clases supera
> el umbral de confianza (apartado 5.1.6). Lo único que las relaciona es
> que las 273 imágenes de `other` son las que sirven para calibrar ese
> umbral, precisamente por ser ajenas a las cinco clases entrenadas.

> La categoría `other` (273 imágenes) se excluyó del entrenamiento al no
> ser una clase semánticamente homogénea sino un conjunto de imágenes
> sin contexto claro: entrenarla como una sexta clase habría introducido
> ruido en el espacio de características aprendido.
>
> Tras excluir `other`, las 4828 imágenes restantes se reparten de forma
> razonablemente equilibrada entre las cinco clases entrenables (906-1034
> imágenes por clase, ratio máximo/mínimo aprox 1,14), lo que hizo
> innecesario ponderar la función de pérdida por clase. El conjunto
> resultante se dividió en particiones de entrenamiento, validación y
> prueba (70 %/15 %/15 %) mediante un `train_test_split` estratificado por
> clase en dos pasos (semilla fija = 42). Se verificó explícitamente que
> las proporciones de clase se mantienen casi idénticas entre
> particiones y que ningún `image_id` aparece en más de una partición.

### Arquitectura del modelo y preprocesado

> El clasificador se construye mediante ajuste fino sobre ResNet18
> preentrenada en ImageNet (`torchvision`), sustituyendo su capa
> totalmente conectada original (1000 salidas) por una capa lineal de
> 512 a 5 unidades, una por clase, seguida de `softmax` en inferencia. La
> elección de un *backbone* convolucional clásico y relativamente ligero
> (11,7 M de parámetros) frente a arquitecturas más recientes o pesadas
> responde a dos factores: (a) el volumen de datos disponible (unas 3400
> imágenes de entrenamiento) es reducido para entrenar arquitecturas de
> mayor capacidad desde cero o incluso para un ajuste fino agresivo, y
> (b) ResNet18 se integra con `pytorch-grad-cam` y presenta una
> estructura por bloques residuales que facilita identificar sin
> ambigüedad la última capa convolucional exigida por el requisito de
> explicabilidad.
>
> Las imágenes se preprocesan de forma determinista para validación y
> prueba ---`Resize(256)` seguido de `CenterCrop(224)`, conversión a tensor y
> normalización con las estadísticas de ImageNet (μ = `[0,485, 0,456, 0,406]`, σ = `[0,229, 0,224, 0,225]`)---, y con *augmentation* adicional
> en entrenamiento ---`RandomResizedCrop(224)` y `RandomHorizontalFlip()`---
> para reducir el riesgo de memorización dado el tamaño limitado del
> conjunto de datos.

### Metodología de entrenamiento: ajuste fino progresivo en dos etapas

> El entrenamiento se estructuró en dos etapas de descongelamiento
> progresivo, una estrategia estándar de ajuste fino que permite
> adaptar el modelo preentrenado al dominio de EVA minimizando el riesgo
> de sobreajuste y el coste computacional.
>
> En la primera etapa se congelan todos los parámetros del *backbone* y se
> entrena únicamente la cabeza (`fc`), 2565 parámetros, con `Adam` (`lr = 1e-3`)
> y `CrossEntropyLoss` sin ponderación por clase (justificado por el
> balance casi uniforme del conjunto de datos), durante 15 épocas y lotes de 64
> imágenes. El mejor `checkpoint` alcanza una exactitud de validación
> (`val_acc`) de 84,28 %.
>
> En la segunda etapa se descongela adicionalmente `layer4`, el último
> bloque residual de ResNet18 (8 396 293 parámetros entrenables en
> total), bajo la hipótesis de que las representaciones más profundas y
> semánticamente específicas de la red ---entrenadas originalmente para
> las 1000 clases de ImageNet--- son las que más se benefician de
> adaptarse al dominio fotográfico-estético de EVA. Las capas más
> tempranas (`layer1`–`layer3`), que capturan patrones genéricos
> de bajo nivel (bordes, texturas, gradientes), pueden en cambio
> mantenerse congeladas sin pérdida de rendimiento. Para evitar que el
> reentrenamiento de `layer4` destruya las características preentrenadas,
> se emplean dos grupos de parámetros con tasas de aprendizaje diferenciadas
> en el mismo optimizador `Adam`: 1e-3 para `fc` y 1e-5 para `layer4`. Esta
> segunda etapa se entrena durante menos épocas que la primera (10
> frente a 15), dado que el número de parámetros entrenables es
> sustancialmente mayor y el conjunto de datos no crece, lo que incrementa el
> riesgo de sobreajuste. El mejor `checkpoint` se obtiene en la época 9 de
> 10, con `val_acc = 85,38 %`. Las curvas de pérdida muestran una
> divergencia leve entre entrenamiento y validación a partir de la época 7
> (`train_loss` sigue bajando mientras `val_loss` se estanca), indicativa de
> un sobreajuste incipiente pero controlado, coherente con el diseño
> deliberadamente conservador de esta etapa. La Tabla 5.2 resume los
> hiperparámetros y el resultado de las dos etapas.

**Tabla 5.2.** Hiperparámetros y `val_acc` de las dos etapas del ajuste
fino progresivo.

  ------------------------------------------------------
  Parámetro        Etapa 1 -- Cabeza  Etapa 2 --
                                      Fine-tuning
                                      ligero
  ---------------- ------------------ ------------------
  Parámetros       fc                 fc+layer4
  descongelados                       

  Parámetros       2565               8 396 293
  entrenables                         

  Optimizador      Adam               Adam

  Learning rate    1e-3(fc)           1e-3(fc) /
                                      1e-5(layer4)

  Función de       CrossEntropyLoss   CrossEntropyLoss
  pérdida                             

  Tamaño de batch  64                 64

  Épocas           15                 10

  Mejor val_acc    84,28 %            85,38 %
  ------------------------------------------------------

### Evaluación sobre el conjunto de prueba

> El resultado de la etapa de ajuste fino ligero se evaluó sobre el
> conjunto de prueba (725 imágenes), la única partición no utilizada en
> ningún momento para seleccionar el modelo, la tasa de aprendizaje o el criterio de
> parada, y que por tanto ofrece una medida no contaminada de
> rendimiento real.
>
> La exactitud global obtenida es del 85,10 %. El desglose por clase
> ---precisión, recall y F1-score---, la matriz de confusión y el
> análisis del error dominante se presentan en el apartado 9.1, junto al
> resto de resultados experimentales.

### Calibración de umbral de confianza para la clase otro

> La clase «otro» no se entrena como una sexta salida del modelo: en su
> lugar se calibra un umbral sobre la probabilidad máxima de `softmax`, y
> toda predicción que no lo supere se rechaza y se reclasifica como
> «otro». A esa etiqueta le corresponde, en el vector W, una fila de
> pesos uniformes que actúa como valor por defecto seguro.
> La calibración se realizó con las 273 imágenes de
> `otro.csv`, nunca vistas durante el entrenamiento ni la selección de
> modelo, comparando la distribución de su confianza máxima con la del
> conjunto de prueba (imágenes que sí pertenecen a una de las cinco
> clases). El valor resultante, el barrido que lo fija y la asimetría de
> coste que justifica apartarse del óptimo puramente cuantitativo se
> presentan en el apartado 9.1.

### Explicabilidad mediante Grad-CAM

> Para satisfacer el requisito de auditabilidad, cada predicción de
> clase del clasificador se acompaña de un mapa de calor `Grad-CAM`
> (Gradient-weighted Class Activation Mapping), generado con la librería
> `pytorch-grad-cam` sobre el último bloque convolucional de `layer4`, la
> capa más profunda y semánticamente específica de la red, por el mismo
> razonamiento empleado para decidir qué bloque descongelar en la etapa
> de ajuste fino. El mapa resultante señala, superpuesto sobre la imagen
> original, las regiones que influyeron en la clase predicha.
>
> Los mapas generados para una imagen representativa de cada clase
> (`outputs/gradcam/`, reproducidos en la Figura 5.1) son compatibles con las categorías descritas: en
> el ejemplo de la
> clase animal, la activación se concentra en la cabeza del ave, la
> región asociada al sujeto, y no en zonas de
> fondo o de plumaje uniforme; en el ejemplo de arquitectura, la
> activación recae sobre la masa de edificios y estructura urbana en el
> plano medio de la imagen, ignorando en gran medida el cielo. Esta
> evidencia visual sirve como herramienta de depuración durante el
> propio desarrollo del modelo y se persiste en disco, de modo que la
> etiqueta de contexto asignada queda auditable a posteriori. No forma parte, en cambio, de los seis paneles que la
> interfaz muestra al usuario final (apartado 7.6.5).
>
> El rechazo a «otro» no genera mapa, y es una decisión deliberada.
> Grad-CAM necesita una clase concreta sobre la que retropropagar, así
> que el único mapa posible en ese caso sería el de la categoría que más
> probabilidad ha reunido, justo cuando esa probabilidad no ha alcanzado
> el umbral y el sistema ha concluido que la imagen probablemente no
> pertenece a ninguna de las cinco. Un mapa así se leería como la
> explicación de un contexto que el clasificador acaba de rechazar, e
> induciría a confiar en una etiqueta que el propio sistema no sostiene.
> No mostrar nada aplica el mismo criterio con el que los especialistas
> del nivel 2 se niegan a publicar como hecho una métrica cuya condición
> de aplicabilidad no se cumple.
>
> ![](docs/memoria/media/media/image6.png){width="5.353472222222222in"
> height="1.1104800962379702in"}
>
> **Figura 5.1.** Mapas `Grad-CAM` sobre una imagen representativa de
> cada una de las cinco clases (animal, producto-still_life,
> arquitectura, paisaje, retrato-humano), superpuestos a la imagen
> original.

### Artefactos finales e integración con el agente orquestador

> El subproyecto se cierra con tres artefactos reutilizables
> directamente por el agente orquestador, mostrados en la Figura 5.2:
> el checkpoint del modelo
> (`fase4_fine_tuning_ligero.pt`), el umbral de rechazo calibrado
> (`umbral.json`, valor 0,85) y el mapeo de índices a etiquetas
> (`class_to_idx.json`). En tiempo de inferencia, el flujo es: (1)
> preprocesar la imagen con `eval_transform`, (2) obtener los logits del
> modelo y aplicar `softmax`, (3) si la probabilidad máxima ≥ 0,85, emitir
> la clase predicha junto con su mapa `Grad-CAM`; en caso contrario,
> emitir «otro», (4) usar la etiqueta resultante para seleccionar la fila
> correspondiente del vector W, que el orquestador transmite al
> agente crítico (nivel 3) para ponderar los informes de los cuatro
> especialistas (nivel 2).
>
> Resolver el contexto con esta CNN, y no con un modelo de lenguaje,
> cumple el segundo requisito de diseño que motivó esa decisión junto
> con la auditabilidad: el coste. La clasificación es una única pasada
> hacia delante sobre una red pequeña, en local y en milisegundos, de
> modo que la etiqueta que gobierna todo el análisis posterior no
> consume ninguna llamada a la interfaz de programación ni introduce
> latencia apreciable. El resto de la percepción del nivel 1 ---la
> detección con YOLO y el mapa de saliencia--- se ejecuta también aquí,
> por el motivo que desarrolla el apartado 5.2: se calcula una sola vez
> y la comparten los cuatro especialistas, en lugar de que cada uno
> repita la misma detección sobre la misma imagen.
>
> ![](docs/memoria/media/media/image7.png){width="2.0937653105861767in"
> height="0.9375065616797901in"}
>
> **Figura 5.2.** Los tres artefactos versionados del clasificador de
> contexto en disco: `pesos_fine_tuning_ligero.pt`, `class_to_idx.json`
> y `umbral.json`.

## Percepción compartida

> La percepción compartida es una herramienta de nivel 1 del orquestador:
> `SharedPerceptionTool` se ejecuta antes de que cualquier especialista
> del nivel 2 empiece a trabajar. Su función es doble, y ninguna de las
> dos partes es opcional: (1) intentar localizar el sujeto principal de
> la fotografía, y (2) garantizar que, si la localización falla o es
> dudosa, los cuatro especialistas reciban la misma información al
> respecto y no cada uno decida por su cuenta si hay o no un objeto
> sobre el que anclar su análisis. Esta segunda parte justifica que la
> percepción compartida se calcule una sola vez y se distribuya, en
> lugar de dejar que cada especialista corra su propia detección: si
> cada agente decidiera por separado dónde está el sujeto, dos
> especialistas podrían analizar bboxes distintos de la misma imagen sin
> que nada lo detectara, y el sistema perdería la propiedad que más le
> interesa: que los cuatro informes hablen del mismo objeto.
>
> Se implementa en `SharedPerceptionTool` / `SharedPerceptionEngine`
> (`tools/shared_perception_tool.py`) y expone una salida de diez campos,
> sincronizada con `schemas/orquestador.py::PercepcionCompartida` y con
> `agents.yaml`/`tasks.yaml`, que la describen al LLM del nivel 1: `bbox`,
> `clase`, `confianza_yolo`, `fuente`, `sujeto_discreto`, `blob_area_ratio`,
> `mapa_saliencia_path`, `centroide_saliencia`, `deteccion_yolo_path` y
> `percepcion_bbox_path`.

### Detección del sujeto con YOLO y umbral de confianza

> La primera fuente candidata para el `bbox` es una detección directa con
> YOLO. El sistema usa `yolo11m.pt` como modelo activo, decisión tomada
> tras evaluar varias opciones (`yolo26n.pt`, `yolo26m.pt`, `yolov8m.pt`,
> `yolov8n.pt`) sobre el corpus de desarrollo y descartarlos. El nombre
> del modelo está escrito en el constructor de `SharedPerceptionEngine`,
> que lo carga una única vez a nivel de módulo para no pagar ese coste
> en cada análisis.
>
> Una detección de YOLO se acepta como `bbox` definido si su confianza
> supera `CONF_YOLO_THRESHOLD = 0.75`. Por debajo de ese umbral se
> descarta y el sistema recurre a la vía de saliencia (apartados 5.2.2 y 5.2.3). El
> `bbox` se publica en formato `[cx, cy, w, h]` (centro y dimensiones),
> junto con la clase COCO detectada y su confianza. Esta convención de
> centro obliga a una conversión explícita más adelante: el Agente 3
> (GrabCut) necesita el rectángulo en formato (x, y, w, h) de esquina,
> que es lo que espera `cv2.grabCut`, mientras que el Agente 1 usa
> `bbox[:2]` directamente como punto de anclaje.
>
> El sistema genera además una imagen de verificación con las
> detecciones de YOLO dibujadas (`deteccion_yolo_path`) independientemente
> de cuál sea la fuente finalmente elegida para el `bbox`.
>
> El umbral de YOLO y de la región conexa de saliencia son heurísticos ajustados sobre
> el corpus de desarrollo ---las 40 imágenes de `data/` que define el apartado
> 8.2---, sin pasar por el ciclo de calibración del apartado 3.3.1, que se
> aplicó a los umbrales del nivel 2.

### Mapa de saliencia por residuo espectral y región conexa de Otsu

> Paralelamente a la detección de YOLO, el sistema calcula siempre ---no
> solo como mecanismo de respaldo--- un mapa de saliencia visual
> mediante el algoritmo de residuo espectral
> (`cv2.saliency.StaticSaliencySpectralResidual`, del paquete
> `opencv-contrib-python`; `cv2.saliency` no está disponible en
> `opencv-python` a secas). Que se calcule siempre y no bajo demanda es
> una decisión deliberada: el mapa no es solo la fuente candidata cuando
> YOLO falla, sino un insumo que consumen directamente otros componentes
> del sistema con independencia de si hay o no un sujeto discreto ---en
> concreto, el Agente 1 lo usa para calcular el equilibrio visual
> (`d_equilibrio`, el centro de masa de energía del mapa completo) y el
> Agente 3 lo usa como referencia de segmentación cuando no hay `bbox` de YOLO
> fiable---.
>
> Cuando YOLO no supera el umbral de confianza, el sistema aplica un
> umbralizado de Otsu sobre el mapa de saliencia para extraer el
> contorno (región conexa) ganador ---el de mayor energía--- y calcula el `bbox`
> como el rectángulo envolvente de ese contorno. Publica junto a él
> blob_area_ratio (la fracción del encuadre que ocupa la región conexa) y
> centroide_saliencia (centro de masa del mapa completo, calculado con
> cv2.moments, independiente de si hay o no una región conexa plausible).
>
> Esa región conexa solo se acepta como `bbox` fiable si su área relativa cae
> dentro de una banda de plausibilidad: BLOB_AREA_MIN_RATIO = 0.005 ≤
> blob_area_ratio ≤ BLOB_AREA_MAX_RATIO = 0.60. Por debajo del mínimo es
> demasiado pequeña para ser un sujeto real (más probablemente ruido de
> saliencia); por encima del máximo ocupa una fracción tan grande del
> encuadre que ya no describe un objeto discreto sino la práctica
> totalidad de la escena.
>
> El mapa de saliencia se persiste siempre en disco
> (`mapa_saliencia_path`), con una restricción para la
> coherencia del sistema: debe conservar las dimensiones exactas de la
> imagen original, porque el Agente 1 deriva de él su espacio de
> coordenadas de trabajo.

### Las tres procedencias del `bbox` y el flag sujeto_discreto

> Con las dos fuentes anteriores, la percepción compartida resuelve el
> `bbox` final por una de tres vías, registradas en el campo fuente:
>
> **Fuente «yolo».** YOLO detectó con confianza ≥ 0,75: `bbox` real,
> con clase COCO e identidad semántica.
>
> **Fuente «saliencia».** YOLO no ofreció nada fiable, pero la región conexa
> ganadora de Otsu es plausible según la banda de área: `bbox` = envolvente
> de ese contorno, sin identidad semántica (solo se sabe que ahí hay
> contraste, no qué es).
>
> **Fuente «sin_sujeto_claro».** Ni YOLO ni la región conexa de saliencia son
> fiables (área fuera de banda, o sin contornos en absoluto): el sistema
> devuelve igualmente un `bbox` de referencia ---centrado en el punto de
> máxima saliencia si no hay ningún contorno utilizable---, pero marcado
> como no fiable.
>
> En los tres casos se emite además el indicador booleano `sujeto_discreto`, y
> este es el diseño central del apartado: los especialistas del nivel 2
> no deben consultar el campo `fuente`, sino este booleano, para decidir
> si existe algo localizable sobre lo que anclar su análisis. La razón
> de introducir un campo derivado en lugar de dejar que cada
> especialista interprete `fuente` por su cuenta es evitar que la lógica
> de «¿hay sujeto o no?» se disperse y posiblemente diverja entre los
> cuatro agentes.
>
> `sujeto_discreto = False` es un hallazgo compositivo legítimo, no un
> error del sistema. Significa «no hay un objeto focal único y
> espacialmente acotado sobre el que poner un rectángulo con
> garantías», que es una categoría real y frecuente en fotografía:
> paisaje de interés distribuido, textura o abstracción, arquitectura
> sin foco claro, bodegones de alta clave donde figura y fondo se
> funden, patrón o multitud. El propio vector W ya asume esta
> posibilidad ---la dimensión de espacio y aislamiento del sujeto vale
> 0,17 en el contexto de arquitectura, el peso más bajo de toda la
> matriz---, así que el sistema no está inventando una categoría nueva:
> está haciendo explícito algo que ya estaba implícito en el diseño de
> W.
>
> Esto conecta con el principio rector de todo el nivel 2, enunciado en
> el apartado 4.4, del que `sujeto_discreto` es la primera
> manifestación. `sujeto_discreto = False` no se
> resuelve saltándose agentes (flujo de control), sino que cada
> especialista sigue midiendo y reporta el resultado con una confianza
> explícita por métrica ---manteniendo ortogonales la importancia
> contextual (que vive solo en W, y el nivel 2 no conoce) y la
> fiabilidad de la medición (que emite cada especialista).
>
> El criterio que conecta este apartado con los capítulos siguientes es
> que la vía de anclaje y la condición de aplicabilidad de confianza
> no dependen de la misma señal. `sujeto_discreto` decide la vía de
> anclaje ---qué punto de referencia usa el especialista: el centro del
> `bbox` si es `True`, el `centroide_saliencia` si es `False`---, pero la condición de aplicabilidad
> que decide si esa medición es citable con confianza plena, en los
> Agentes 1 y 3, no cuelga de `sujeto_discreto` sino de `fuente ==
> "yolo"`, que es más estricta. El fundamento es semántico: YOLO aporta
> identidad ---si dice «persona»
> o «árbol», hay un objeto real y con nombre---, mientras que la región conexa
> de saliencia solo dice «aquí hay contraste», y eso lo produce igual
> un patrón de textura, un muro o una masa de vegetación que un sujeto
> real. Con ese criterio, la condición cierra en torno a la
> mitad del corpus, y dos especialistas que antes podían contradecirse
> sobre la misma evidencia (uno afirmando dónde está el sujeto, otro
> negando que exista un objeto que aislar) pasan a estar de acuerdo.
>
> Finalmente, la percepción compartida deja evidencia visual verificable
> de su propia decisión: además de la imagen con las detecciones de YOLO
> (apartado 5.2.1), genera una imagen de verificación con el `bbox` final ya
> resuelto, coloreada según su fuente (`percepcion_bbox_path`).

## Salida estructurada del orquestador y evidencia visual generada

> Cerrado el clasificador de contexto y la percepción compartida, el
> nivel 1 empaqueta ambos resultados en una única salida tipada:
> `SalidaOrquestador` (Pydantic, `schemas/orquestador.py`) con la forma
> `{etiqueta_contexto, percepcion_compartida}`, y esta segunda a la vez en
> diez campos descritos en el apartado 5.2. Se conecta en la tarea del agente
> orquestador mediante `output_pydantic`, de modo que esa tarea no
> entrega texto libre sino un objeto validado por esquema antes de que
> ningún especialista lo consuma.
>
> La evidencia visual que deja el nivel 1, previa a que actúe ningún
> especialista, se compone de hasta cuatro imágenes por análisis, todas
> generadas antes de que empiece el nivel 2 y disponibles para
> inspección directa ---en el desarrollo, y más tarde en los paneles de
> verificación de la interfaz---:
>
> El mapa de calor de `Grad-CAM` sobre la predicción del clasificador de
> contexto (apartado 5.1.7), que explica en qué regiones se apoyó la CNN para
> decidir la etiqueta. Es el único de los cuatro que puede faltar: cuando
> el contexto se resuelve como «otro» no se genera, por el motivo que
> explica ese mismo apartado.
>
> La imagen con las detecciones de YOLO dibujadas ---`bbox`, clase y
> confianza--- con independencia de si esa detección terminó siendo la
> fuente elegida para el `bbox` final.
>
> El mapa de saliencia por residuo espectral, persistido tal cual lo
> produce el algoritmo.
>
> El `bbox` final ya resuelto, coloreado según su procedencia (`yolo` /
> `saliencia` / `sin_sujeto_claro`), que es la imagen que certifica
> visualmente la decisión descrita en el apartado 5.2.3.
>
> Todas ellas salvo el mapa de saliencia se reducen a 1600
> px de lado mayor al escribirse en disco, con una compresión PNG más
> agresiva: es una optimización pensada para la interfaz, que reduce
> drásticamente el peso de las carpetas de salida sin perder legibilidad
> (las carpetas shared_perception/ y yolo/ pasaron a ocupar en torno a
> una quinta parte de lo que ocupaban). El mapa de
> saliencia queda deliberadamente excluido de esa reducción: los Agentes
> 1 y 3 lo vuelven a leer de disco en su resolución original ---el
> primero para fijar su espacio de coordenadas de trabajo, el segundo
> como referencia de segmentación cuando no hay `bbox` de YOLO fiable---, así
> que reducirlo introduciría un desajuste silencioso entre el espacio de
> coordenadas del mapa y el de la imagen original.
>
> Con esto, el nivel 1 entrega a los cuatro especialistas exactamente
> dos cosas: una etiqueta de contexto y una percepción
> compartida con su `bbox`, su indicador booleano de aplicabilidad y su evidencia visual
> ---nunca el vector de pesos, que solo aparece en el nivel 3, donde se
> decide cuánto pesa cada dimensión.

# Nivel 2 -- Especialistas y métricas verificables

## Patrón común de un especialista

> Los cuatro especialistas (composición espacial, líneas y dirección, espacio y
> aislamiento del sujeto, luz y tono) miden una dimensión compositiva distinta y
> reducen al mínimo el solape entre ellas, pero comparten una arquitectura interna.
> Documentar este patrón antes de cada agente permite comparar los apartados 6.2 a
> 6.5 entre sí.

### La herramienta

> Cada especialista es una `BaseTool` de CrewAI que envuelve un motor en
> Python puro (`NumPy`/`OpenCV`/`scikit-learn`). La herramienta no razona:
> recibe la imagen y los datos de percepción compartida que le corresponden,
> invoca al motor y devuelve `model_dump()` como JSON limpio para que el LLM lo
> lea sin ambigüedad. Los Agentes 1 y 3, que consumen la percepción compartida (con
> `context=[compose_task]`), reciben ese bloque, mientras que los dos puramente
> globales reciben solo `ruta_imagen` (Agentes 2 y 4, con `context=[]`).

### El esquema (doble autoría)

> Cada especialista emite `{informe, diagnóstico}`, con autorías separadas. Este
> es el punto central de la sección:

- **Informe.** Lo produce el motor como un modelo Pydantic y es
  recomputable por un tercero. Es la evidencia auditable.

- **Diagnóstico.** Lo redacta el LLM en 4-6 frases, citando el informe
  con notación de punto (`[d_tercios.valor = 0.0648]`), nunca
  transcribiéndolo ni repitiéndolo sin más.

> El apartado 4.3.1 define el envoltorio `MetricaConfianza`; aquí importa su
> convención de lectura. `valor_norm` pertenece a `[0 -- 1]`, con `1 = adherencia
> máxima`, y vale `None` cuando la métrica carece de una cota de adherencia
> interpretable, como d_equilibrio, o es una categoría y no una magnitud, como
> patron_dominante. Los campos confianza y fuente_confianza expresan la condición
> de aplicabilidad.
>
> Esta separación es lo que permite medir en el capítulo 9 dos
> fidelidades distintas: fidelidad de transcripción (¿el LLM copia el
> informe sin inventar, omitir o corromper un campo?) y fidelidad
> métrica-texto (¿toda cifra que aparece en prosa tiene respaldo en el
> informe?).

### La condición de aplicabilidad

> Como establece el apartado 4.4, cada métrica se mide y publica siempre; la
> condición de aplicabilidad decide si es citable. Esta condición reúne los modos
> de fallo de las señales concretas de las que depende cada métrica, por eso el
> número de condiciones varía.

### El diagnóstico (control de la redacción)

> Los cuatro especialistas comparten el mismo contrato de redacción, que
> el apartado 4.5.3 enuncia entero: toda afirmación va citada sobre un
> escalar, está prohibido hablar de importancia y usar adjetivos
> valorativos, y los adverbios de grado exigen una escala documentada.
> Lo que interesa retener aquí, como parte del patrón, es la forma que
> acabó tomando ese contrato: una comprobación final de criterios
> con la que el modelo relee su propio borrador y borra las frases que
> los incumplen, en vez de una lista creciente de prohibiciones. El
> apartado 6.6 cuenta por qué esa forma funciona mejor y qué la hizo
> necesaria.

### El panel de verificación visual

> Los cuatro reutilizan `CapaVerificacion`:
> primitivas comunes (línea, punto, rectángulo, etiqueta), grosor y
> escala de texto derivados de `min(w,h)` para ser legibles en cualquier
> resolución, y la convención cromática compartida de todo el sistema
> ---verde/rojo codificando la confianza---. Cada agente decide qué
> dibujar (rejilla de tercios, segmento de horizonte, contorno de
> máscara, franja de matices dominantes), pero cómo dibujarlo es
> infraestructura común. Los paneles forman parte del contrato de trazabilidad y
> la interfaz Streamlit los muestra junto a la síntesis; no son un añadido
> cosmético.

## Agente 1 -- Composición espacial

> El agente 1 estudia dos fenómenos compositivos independientes. Sitúa el sujeto
> respecto a dos referencias clásicas rivales y añade una medición global del
> peso visual del encuadre. Es también el agente donde el sistema fijó
> por primera vez la convención de normalización que heredan las otras
> nueve métricas (apartado 6.1).

### Qué mide y qué no mide

> **Regla de los tercios.** La métrica `d_tercios` mide la distancia del
> anclaje del sujeto al punto de fuerza más cercano de la rejilla de
> tercios, las cuatro intersecciones normalizada por la diagonal de la
> imagen. El anclaje es el centro del `bbox` si `sujeto_discreto = True`, o
> el centroide del mapa de saliencia si es `False`; qué vía se usa es
> decisión de su condición de aplicabilidad en el apartado 6.2.4.
>
> **Centrado.** `d_centro` existe como hipótesis rival, con el mismo
> anclaje al `bbox` y mismo denominador, es la distancia al centro
> geométrico. Existe para resolver una ambigüedad real: una composición
> centrada deliberada (retrato frontal, bodegón, simetría) puntúa mal en
> la regla de los tercios sin que eso sea un fallo compositivo. El patrón
> al que se acoge la fotografía es el de menor distancia entre las dos.
>
> **Equilibrio visual.** La métrica `d_equilibrio` mide la distancia entre
> el centro de masa de todo el mapa de saliencia (via `cv2.moments`,
> `m10/m00`, `m01/m00`) y el centro geométrico de la imagen. A diferencia de
> las dos métricas anteriores (d_tercios, d_centro), no depende de ningún
> anclaje al sujeto. Mide el reparto del peso visual de la escena
> completa. Por eso, la confianza es siempre 1. Es la única métrica del
> agente sin condición de aplicabilidad y su motivo se explica en el apartado 6.2.4.

### Cálculo, entradas, salidas e implementación

> La implementación se encuentra en
> `src/tfg_multiagente_fotografia/tools/composicion_espacial_tool.py`.
> Recibe el bloque de percepción compartida, utiliza como anclaje el
> centro del `bbox` o el centroide del mapa de saliencia y devuelve las
> métricas `d_tercios`, `d_centro`, `patron_dominante` y
> `d_equilibrio`, junto con sus campos de confianza.
>
> **Arbitraje determinista.** La decisión
> de a qué patrón (tercios o centrado) se acoge la foto no la toma el
> LLM: la calcula el motor como `margen_patron = |d_tercios - d_centro|`, y
> gana el patrón de menor distancia.
>
> El motor aplica guardas de robustez: lanza `ValueError` si el mapa de
> saliencia no se puede leer o si m00 = 0 (mapa sin energía); se
> prefiere fallar explícitamente a inventar un equilibrio perfecto sobre
> un centro de masa indefinido.

### Alternativas descartadas y decisiones de diseño

> Se decidió normalizar `d_tercios`, `d_centro` y `d_equilibrio` por la misma
> magnitud, la diagonal, en vez de que cada una normalice por su propia
> cota. Esta decisión permite compararlas; si se normalizaran por separado,
> comparar «cuál de las dos referencias está más cerca» perdería sentido
> matemático.

### Condiciones de aplicabilidad y calibración

> Inicialmente, la condición de aplicabilidad de `d_tercios`, `d_centro` y
> `patron_dominante`
> colgaba de `sujeto_discreto`, la señal binaria heredada directamente de
> la percepción compartida (nivel 1).
>
> La decisión adoptada fue obtener la condición de aplicabilidad con
> `fuente == "yolo"`, el criterio ya calibrado del Agente 3. YOLO
> aporta identidad de clase, la región conexa de saliencia solo dice «aquí hay
> contraste». La revisión visual que documenta el apartado 6.2.5 cifró el
> balance de la decisión en 16 falsos positivos eliminados frente a 3
> falsos negativos introducidos. Estos últimos proceden de clases ausentes de
> YOLO, como árbol o edificio; el apartado 6.4.6 desarrolla esta limitación y el
> 9.5 la recoge entre las limitaciones transversales.

### Mediciones sobre el corpus de desarrollo

> El síntoma se cuantificó así: la condición de aplicabilidad dejaba pasar con confianza plena al
> 97,5 % del corpus de desarrollo (39 de 40 imágenes) --- el mismo
> diagnóstico que ya se había hecho sobre la condición de aplicabilidad del horizonte del
> Agente 2 cuando dejaba pasar al 90 %. Se aplicó el criterio, ya empleado
> entonces, de que una condición de aplicabilidad que casi nunca cierra
> probablemente no esté midiendo aplicabilidad, sino que la percepción
> compartida devolvió algo.
>
> La recalibración aplicó el ciclo del apartado 3.3.1 mediante una revisión visual
> manual de las 19 imágenes con `fuente == "saliencia"`, el subconjunto cuyo
> `bbox` procede de la región conexa de contraste de Otsu y no de una detección
> YOLO fiable. Solo 3 de 19
> tenían un sujeto real correctamente delimitado. Los 16 restantes se
> repartían en las cuatro familias de falso positivo de la Tabla 6.1:

**Tabla 6.1.** Familias de falso positivo entre las 19 imágenes con
`fuente == "saliencia"`, revisadas visualmente.

  ------------------------------------------------------
  Familia             n               Ejemplo
  ------------------- --------------- ------------------
  Interés distribuido 8               Escenas sin foco
  (arquitectura,                      único
  paisaje)                            

  Anclaje en la zona  4               Espuma de una ola,
  más brillante, no                   reflejo del sol en
  en el interés real                  asfalto

  Patrón/Textura      2               Tejido, muro

  No fotografías      2               Captura de
                                      pantalla, pintura
                                      abstracta
  ------------------------------------------------------

> El segundo subgrupo concentra el fallo en el que el sistema no solo fallaba,
> sino que afirmaba activamente una localización incorrecta con confianza plena.
>
> En las 40 imágenes del corpus de desarrollo, las fuentes fueron `yolo` en 20,
> `saliencia` en 19 y `sin_sujeto_claro` en 1. La condición de aplicabilidad se
> cerró en 20 de 40 imágenes (50 %), frente a 1 de 40 (2,5 %) con la condición
> de aplicabilidad original.

### Alcance y límites de interpretación

> La métrica sin cota interpretable requiere una consideración específica. La
> cota de `d_equilibrio` es ½, igual que `d_centro`, pero el
> rango realmente observado es más estrecho, porque el algoritmo de
> residuo espectral reparte la energía por todo el fotograma y arrastra
> el centro de masa hacia el centro geométrico. Se deja para trabajo
> futuro buscar una escala normalizada suficientemente defendible.
> Mientras tanto, `valor_norm = None` y la plantilla de instrucciones usa referencias
> orientativas medidas en el corpus (< 0,03 equilibrada, > 0,12
> descompensada), que no son umbral de aplicabilidad.
>
> El trabajo futuro relacionado con los apartados 9.5 y 10.4 consiste en
> sustituir la cota teórica por una normalización empírica por percentiles sobre
> el corpus (EVA o el propio `data/`). Esta modificación no obliga a tocar el
> esquema: solo cambia cómo se calcula `valor_norm`.

## Agente 2 -- Líneas y dirección

> El agente 2 mide dos fenómenos geométricos independientes sobre los segmentos
> de recta que la imagen ofrece: cuánto se desvía de la horizontal la línea que
> funciona como horizonte, y hasta qué punto las líneas no horizontales
> concurren en un punto de fuga común. Es, junto al agente 4, puramente global:
> no consume `bbox`, ni `centroide_saliencia`, ni `sujeto_discreto`, y su única
> entrada es `ruta_imagen`. Tiene dos condiciones de aplicabilidad independientes,
> una por métrica, y necesita un artefacto calibrado fuera de línea ---el modelo nulo--- para
> decidir si lo que ha encontrado es perspectiva o azar.

### Qué mide y qué no mide

> **Horizonte.**
> [PENDIENTE-AUTOR: explicar qué miden `angulo_horizonte` y
> `longitud_horizonte`, incluida la convención del signo, y qué no
> permite afirmar la presencia de un segmento casi horizontal.]
>
> **Convergencia.** La métrica `score_convergencia` mide el
> ratio de inliers del punto de fuga ganador, estimado por RANSAC sobre
> las intersecciones de segmentos de franja de fuga. La pregunta
> compositiva es «¿tiene esta foto una estructura de perspectiva marcada
> que organice la mirada en profundidad?», no «¿los elementos lineales
> señalan al sujeto?». Esta lectura es la que sostiene el vector W: la
> dimensión de líneas vale 0,37 en arquitectura, un peso que destaca
> frente a otros contextos y que Präkel justifica por las líneas
> convergentes como elemento compositivo central en esa categoría.
>
> Un campo nuevo que resuelve una limitación por construcción del score:
> apertura_haz. score_convergencia no puede distinguir convergencia de
> paralelismo, porque un haz de rectas paralelas también «concurre» en
> un punto lejano y todas sus líneas lo votan --- el score es
> matemáticamente invariante en el límite de haz paralelo. apertura_haz
> (diámetro angular del haz de inliers, en grados) sí lo es, y se
> publica porque forma parte de la condición de aplicabilidad: sin ella el crítico no podría
> auditar por qué se cerró.

### Cálculo, entradas, salidas e implementación

> La implementación se encuentra en
> `src/tfg_multiagente_fotografia/tools/lineas_direccion_tool.py`.
> Recibe `ruta_imagen`, detecta y clasifica los segmentos y devuelve
> las métricas de horizonte y convergencia junto con los campos que
> permiten auditar sus condiciones de aplicabilidad.
>
> **Horizonte.**
> [PENDIENTE-AUTOR: describir cómo se elige el segmento candidato, cómo
> se calculan `angulo_horizonte` y `longitud_horizonte`, en qué
> unidad o escala se publican y qué campos forman la salida.]
>
> **Convergencia.** Una condición previa basada en un modelo nulo evita
> interpretar como perspectiva el mejor patrón producido por azar. El resultado
> observado se compara con el de líneas equivalentes cuya estructura de
> convergencia se ha aleatorizado. El ratio de inliers solo constituye evidencia
> de perspectiva cuando supera ese consenso accidental; en caso contrario, el
> valor no llega al crítico.
>
> Una vez superada esta validación, queda por determinar el mejor punto
> de convergencia. Aunque RANSAC utiliza habitualmente un muestreo
> aleatorio de hipótesis, aquí se enumeran todas: la hipótesis mínima
> queda definida por dos rectas, así que para (n) líneas existen
> únicamente (C(n,2)=n(n-1)/2) parejas posibles, una complejidad
> (`O(n^2)`) suficientemente reducida para los tamaños de entrada del
> sistema y con un coste de apenas milisegundos en el peor caso
> observado. El apartado 6.3.3 desarrolla por qué se prefiere esa vía a
> la habitual.
>
> Las dos piezas son complementarias: el modelo nulo descarta convergencias
> explicables por azar y la búsqueda exhaustiva hace reproducible el resultado.
> Así, score_convergencia solo se incorpora al análisis cuando existe evidencia
> suficiente de una estructura perspectiva y puede calcularse de manera
> determinista.
>
> **Reproducibilidad.** El modelo nulo está acoplado al estimador y la
> guarda es automática. parametros/modelo_nulo_fuga.json se genera
> fuera de línea sobre escenas sintéticas y depende de las constantes exactas
> del núcleo de consenso; tocar cualquiera de ellas invalida la tabla.
> Desde la puesta en producción, cargar el modelo nulo compara los
> parámetros guardados contra las constantes vivas del módulo y lanza
> una excepción al importar si divergen --- convierte lo que en otro
> sistema sería una deuda silenciosa en una guarda estructural, y es un
> ejemplo citable en el apartado 8.3.2.

### Alternativas descartadas y decisiones de diseño

> **Horizonte.**
> [PENDIENTE-AUTOR: indicar si se comparó el cálculo final del horizonte
> con alguna alternativa; si no hubo alternativa, declararlo
> expresamente.]
>
> **Convergencia.** La alternativa que se descarta es el muestreo
> aleatorio de hipótesis, que es la forma habitual de aplicar RANSAC. El
> motivo de apartarse de ella no es una preferencia, sino una condición
> del problema que el propio artículo que introduce el método contempla.
> Fischler y Bolles [9] plantean el sorteo como respuesta al coste de
> recorrer los subconjuntos mínimos ---por eso dedican un apartado a
> acotar el número esperado de intentos, que crece con el tamaño de esa
> muestra mínima--- y señalan expresamente que, cuando existe un criterio
> del problema para elegir esos subconjuntos, conviene sustituir la
> selección aleatoria por una determinista.
>
> Aquí se cumplen las dos condiciones. La muestra mínima son dos rectas,
> el menor valor posible, de modo que el conjunto completo de hipótesis
> se reduce a las (C(n,2)) parejas y recorrerlo entero cuesta
> milisegundos incluso en la imagen con más líneas del corpus. Y hay un
> criterio del problema para preferir ese recorrido completo: enumerar
> todas las parejas devuelve por construcción la hipótesis más votada de
> todas, mientras que un sorteo devuelve la mejor de las que le tocaron.
>
> La elección no la decide la precisión, sino la reproducibilidad.
> score_convergencia no es un
> valor intermedio del algoritmo, sino una observación cuantitativa que
> el agente crítico puede citar para sostener una afirmación. Con
> muestreo, ese número depende de la semilla y del número de
> iteraciones, dos parámetros arbitrarios que habría que justificar y
> que harían que una misma fotografía pudiera recibir puntuaciones
> distintas en dos ejecuciones. Al enumerar, esos dos parámetros
> sencillamente no existen, y la métrica pasa a ser recomputable por un
> tercero, que es la premisa sobre la que se apoya el resto del sistema.
>
> Se probaron además dos ajustes más conservadores. Elevar el modelo nulo
> del percentil 95 al 99 cerró tres falsos positivos (`imagen24`,
> `imagen3` e `imagen11`), aunque también cerró `imagen13`, una fachada
> real sostenida por pocos segmentos. Estrechar de 15° a 10° la
> partición entre horizonte y fuga tampoco eliminó los cuatro falsos
> horizontes y elevó de 4 a 10 las imágenes que abrían la condición de aplicabilidad de
> convergencia, por lo que se descartó.
>
> Las verticales quedan fuera de la franja de fuga, y la decisión se apoya
> en una medición y no solo en la intuición: incluirlas abre 3 condiciones de aplicabilidad adicionales, y la
> revisión visual de esas tres imágenes (retrato, producto y fauna) las
> identificó como falsos positivos --- cero verdaderos positivos ganados.

### Condiciones de aplicabilidad y calibración

> **Horizonte.** Las dos bandas vacías que sostienen los dos umbrales,
> medidas sobre las 40 imágenes del corpus de desarrollo: la de la
> longitud separa 0,1282 de 0,1398 ---0,0116 de margen, y la que cierra
> por poco es imagen30---, y la del ángulo separa 7,97° de 10,66°, con
> 2,69° de margen.
> [PENDIENTE-AUTOR: desarrollar el paso de la condición de aplicabilidad de una a tres
> condiciones; explicar el origen de la existencia de candidato, de
> `longitud_horizonte >= 0.13` y de `|angulo_horizonte| <= 10°`; y
> documentar la revisión visual del patrón de un número, el borde de una
> colina y una línea de fuga sin horizonte.]
>
> **Convergencia.** Tras la revisión visual, la condición de aplicabilidad pasó
> de dos criterios a tres, como la del horizonte:
>
> - **Existencia.** Debe haber al menos 2 candidatas.
>
> - **Ausencia de azar.** `score_convergencia` debe ser mayor que
>   `score_convergencia_nulo` (comparación estricta contra la tabla del modelo nulo).
>
> - **Ausencia de paralelismo.** `apertura_haz` debe ser ≥ 35°.
>
> Sin la tercera condición, abrían la condición de aplicabilidad un muro de textura, una masa
> de vegetación y una escena de fauna --- todos haces de rectas casi
> paralelas que el modelo nulo no caza, porque su pregunta es «¿esto es
> más coherente que el azar?» y un haz paralelo es, precisamente, más
> ordenado que el azar, no menos. La separación empírica es limpia:
> sobre las 25 imágenes con las que se calibró, convergencia real en
> 42°--90° de apertura y paralelismo en 9°--30°, con el corte en 35°
> cayendo en la banda vacía entre ambos.

### Mediciones sobre el corpus de desarrollo

> **Horizonte.** Sobre las 40 imágenes del corpus de desarrollo, la condición de aplicabilidad
> queda abierta en 16 y cerrada en 24. De las que abren, la longitud del
> segmento elegido va de 0,1398 a 0,6362 de la diagonal, con media
> 0,2896, y el ángulo absoluto no pasa de 7,97°. La comparación con la
> condición de aplicabilidad inicial da la medida de lo que aportaron las dos condiciones
> nuevas: con la sola condición de existencia se afirmaba una nivelación
> en 36 de las 40 imágenes ---el 90 % del corpus---, y ahora se afirma
> en 16.
>
> **Convergencia.** Resultado de la calibración original, sobre las 25
> imágenes que tenía entonces el corpus: cero falsos positivos en las
> categorías sin perspectiva, que son donde un estimador de punto de
> fuga es más propenso a afirmar una estructura que no existe. Abren la
> condición de aplicabilidad 4 imágenes y las 4 son perspectiva real verificada a ojo (una galería,
> un pasillo, una fachada, una carretera al horizonte). Retrato 0 de 2,
> producto 0 de 3, paisaje 0 de 8. Sobre las 40 imágenes del corpus de
> desarrollo la condición de aplicabilidad abre en 7, y la fila de producto ha dejado de
> valer: una de las fotografías añadidas después, imagen32, es el primer
> falso positivo confirmado de esta condición de aplicabilidad y se analiza en el apartado
> 9.2.1.
>
> **Validación de extremo a extremo en los dos extremos.** La transcripción del
> informe fue exacta en el 100 % de los campos tanto con ambas condiciones de
> aplicabilidad abiertas como con ambas cerradas (0 segmentos detectados). Las dos
> iteraciones de plantilla de instrucciones se debieron únicamente al caso
> cerrado: un doble escapado de la ruta de verificación en Windows y la costumbre
> de citar fuente_confianza copiando la frase
> completa dentro de los corchetes de cita, lo que además arrastraba a
> la prosa los umbrales numéricos de las condiciones de aplicabilidad (15°, 2 líneas) como si
> fueran campos del informe. Es la misma lección que ya dejó el Agente
> 1: validar el camino en que la métrica no se puede medir es más
> informativo que validar solo el camino feliz.

### Alcance y límites de interpretación

> **Horizonte.**
> [PENDIENTE-AUTOR: explicar la resolución angular, la sensibilidad a
> patrones casi horizontales y por qué detectar un candidato no permite
> afirmar por sí solo que exista un horizonte compositivo.]
>
> **Convergencia.** Encontrar un consenso no demuestra que exista
> perspectiva. RANSAC busca el punto hacia el que parecen converger más
> líneas y, con suficientes líneas, casi siempre encuentra alguna
> combinación que produce cierto consenso aunque el patrón sea
> accidental: en un retrato sobre fondo liso, los contornos de los
> hombros, el pelo o la ropa pueden cruzarse de forma casual y sostener
> un punto aparentemente respaldado. Por eso score_convergencia no
> afirma por sí solo que la escena tenga una estructura de perspectiva
> real; solo lo afirma cuando además su condición de aplicabilidad está abierta. Es el mismo
> límite que el Agente 3 encuentra en cv2.grabCut, un algoritmo diseñado
> para devolver siempre una segmentación aunque la imagen no ofrezca una
> separación figura-fondo que la haga significativa.

## Agente 3 -- Espacio y aislamiento del sujeto

> El agente 3 utiliza dos métricas, cinco campos de evidencia y, a diferencia del
> Agente 2, ningún artefacto fuera de línea. Su condición de aplicabilidad tiene
> un solo criterio y consume la percepción compartida como requisito para que
> exista la medición.
>
> Es, además, el único de los cuatro que cambia de nombre respecto al
> anteproyecto, donde se llamaba «espacio y profundidad» y se le
> encomendaba, junto al espacio negativo y al aislamiento del sujeto,
> una jerarquía de planos. Esa tercera métrica no se ha implementado y
> el agente pasa a llamarse por lo que mide de verdad: el nombre
> anterior prometía una lectura en profundidad que el sistema no puede
> sostener sin estimarla con otro modelo, decisión que se discute en el
> apartado 6.4.3.

### Qué mide y qué no mide

> **Espacio negativo.** ratio_espacio_negativo: la fracción del encuadre
> no ocupada por la máscara del sujeto, en `[0 -- 1]` (0 = el sujeto
> llena el fotograma, 1 = no ocupa nada). valor_norm = None, dado que
> ese campo significa el grado de adherencia a un patrón y más espacio
> negativo no es más adherente a nada: un primer plano que llena el
> encuadre y un sujeto pequeño en un vacío amplio son dos decisiones
> compositivas legítimas, no una correcta y otra incorrecta.
>
> **Separación figura-fondo.** El resultado de ratio_nitidez queda
> acotado en `[-1, 1]` y su signo tiene una interpretación directa:
> valores positivos indican mayor nitidez en la figura, negativos mayor
> nitidez en el fondo y 0 ausencia de diferencia. Para su representación
> normalizada se utiliza valor_norm = (valor + 1) / 2. El único caso
> degenerado que debe tratarse explícitamente ocurre cuando ambas
> varianzas son cero.

### Cálculo, entradas, salidas e implementación

> La implementación se encuentra en
> `src/tfg_multiagente_fotografia/tools/espacio_aislamiento_tool.py`.
> Recibe la imagen y el bloque de percepción compartida, construye una
> máscara del sujeto y devuelve `ratio_espacio_negativo`,
> `ratio_nitidez`, las varianzas originales y la procedencia de la
> máscara, junto con sus campos de confianza.
>
> La implementación ofrece una interfaz uniforme y dos vías de procesamiento,
> seleccionadas por `sujeto_discreto`.

- `sujeto_discreto = True`: se utiliza GrabCut, porque indica que hay un `bbox`
  fiable. El
  `bbox` de la percepción compartida llega como `[cx, cy, w, h]`
  (centrado) y cv2.grabCut exige (x, y, w, h) de esquina superior
  izquierda, por lo que hay que convertirlo.

- `sujeto_discreto = False`: se utiliza un prior de saliencia. No hay un objeto
  delimitable mediante `bbox` y no se puede inicializar GrabCut de forma
  fiable. La máscara se obtiene con Otsu sobre el mapa de saliencia que
  la percepción compartida ya guardó en disco, sin dependencias nuevas.

> El campo fuente_mascara (grabcut/prior_saliencia) se publica siempre,
> porque sin él el crítico no podría auditar de dónde sale la confianza
> de cada métrica.
>
> Un índice normalizado puede ocultar las magnitudes que lo originan.
> Para mantener la trazabilidad, el sistema publica también
> var_laplaciano_figura y var_laplaciano_fondo en sus valores
> originales. De este modo, un tercero puede comprobar la condición de aplicabilidad y
> recomputar ratio_nitidez de forma independiente.
>
> Dos decisiones de implementación específicas de esta métrica, y ambas
> motivadas por cómo funciona el operador de Laplaciano:
>
> Espacio de trabajo fijo de 1024 px, por un motivo distinto al de los
> Agentes 2 y 4: la varianza del Laplaciano depende de la resolución (a
> más píxeles, más bordes finos que el operador recoge), así que sin un
> tamaño de trabajo uniforme, VAR_LAPLACIANO_MIN significaría cosas
> distintas en una foto de 800 px y en una de 4096.
>
> Se descuenta la frontera figura-fondo antes de medir la nitidez. El
> Laplaciano se dispara justo en el borde entre las dos regiones ---es
> literalmente un salto de intensidad, lo que el operador detecta---,
> así que ese borde infla las dos varianzas a la vez y comprime el
> índice hacia 0. Se erosionan ambas máscaras un margen proporcional al
> lado menor de la imagen; si esa erosión vaciara por completo una
> región (sujeto minúsculo), se mide sin ella y se marca.

### Alternativas descartadas y decisiones de diseño

> **Los dos modelos de visión que el anteproyecto proponía para esta
> dimensión.** El planteamiento inicial contemplaba SAM para la
> segmentación del sujeto y MiDaS para la estimación de profundidad.
> Ninguno de los dos se ha utilizado, y la razón es la misma en los dos
> casos: el coste de integrarlos y de mantenerlos no se compensaba con
> lo que aportaban a las dos métricas que esta dimensión necesita medir.
>
> En el caso de SAM, la máscara que produce es mejor que la de GrabCut,
> pero el agente no necesita una máscara mejor: necesita saber si hay un
> sujeto que aislar, y eso SAM no lo responde, porque segmenta lo que se
> le señale sin aportar identidad semántica. Quien la aporta es YOLO, y
> es de la procedencia del `bbox` ---no de la calidad de la máscara--- de
> quien depende la condición de aplicabilidad de este agente. GrabCut inicializado con ese
> mismo `bbox` da una segmentación suficiente para calcular la fracción de
> encuadre ocupada y comparar la densidad de detalle de las dos
> regiones, que es todo lo que se publica, y evita cargar un segundo
> modelo pesado que competiría por la misma GPU.
>
> En el caso de MiDaS hay además un motivo que afecta al contrato del
> nivel 2. Un mapa de profundidad monocular es la salida de otra red
> neuronal, sin escala absoluta y sin una referencia con la que
> comprobarlo, de modo que una métrica derivada de él no sería
> recomputable por un tercero: el crítico acabaría citando una
> profundidad que nadie puede verificar, que es justamente lo que el
> resto del sistema evita. La relación figura-fondo que la profundidad
> pretendía capturar se aborda por una vía sí medible, ratio_nitidez,
> con sus dos varianzas publicadas al lado y con su límite declarado en
> el apartado 9.5.
>
> También se descartaron tres condiciones previstas para la condición de aplicabilidad. La
> concentración de saliencia recorrió de 0,032 a 0,539 sin una banda que
> permitiera fijar un corte; `COBERTURA_MIN` no separaba una segmentación
> correcta de 0,003 (`imagen9`) de una fallida de 0,011 (`imagen14`); y
> `COBERTURA_MAX` carecía de casos de calibración, pues el máximo
> observado, 0,500 en `imagen17`, no era una degeneración de GrabCut. En
> su lugar, la aplicabilidad se vinculó a que la máscara procediera de
> una detección semántica de YOLO.
>
> **Separación figura-fondo.** La formulación original de ratio_nitidez
> como var_figura / var_fondo presentaba una limitación: no está acotada
> y se vuelve problemática cuando la varianza del fondo se aproxima a
> cero. Para evitar esta dependencia se sustituyó por:
>
> ratio_nitidez = (var_figura − var_fondo) / (var_figura + var_fondo)
>
> Otra limitación que se quiso evitar fue hacer depender la
> interpretación de la métrica del corpus utilizado durante el
> desarrollo. Por ello, ratio_espacio_negativo se interpreta
> directamente como proporción del encuadre y ratio_nitidez puede
> traducirse analíticamente al cociente original mediante
> var_figura/var_fondo = (1+v)/(1−v). Así, por ejemplo, v=0,33, 0,50 y
> 0,80 equivalen aproximadamente a una figura con 2, 3 y 9 veces la
> varianza del fondo. Estas referencias no necesitan recalibrarse al
> aumentar el corpus. En cambio, VAR_LAPLACIANO_MIN sí es un umbral
> obtenido empíricamente y, por tanto, constituye una dependencia del
> corpus que se documenta por separado.
>
> **Transformación del `bbox`.** La comparación entre la implementación de
> prueba y la de producción permitió detectar otra fuente de
> discrepancias: el modo en el que se transformaban las coordenadas del
> `bbox` al espacio de trabajo. La implementación de producción utilizaba
> el factor de escala teórico,
>
> 1024 / lado_mayor,
>
> mientras que la sonda de calibración empleaba la escala efectiva resultante después
> del redondeo realizado por cv2.resize. En una imagen grande, por
> ejemplo, ambos factores eran 0,250000 y 0,250076. Aunque la diferencia
> parece insignificante y supone menos de un píxel en las coordenadas
> del rectángulo, era suficiente para que dos implementaciones que
> conceptualmente ejecutaban el mismo procedimiento no produjeran
> resultados idénticos bit a bit.
>
> La solución adoptada fue calcular la transformación a partir de las
> dimensiones reales de la imagen redimensionada y hacerlo
> independientemente para cada eje, eliminando así la dependencia del
> factor de escala teórico. Además, reescalar() dejó deliberadamente de
> devolver dicho factor, evitando que otras partes del sistema puedan
> reutilizarlo accidentalmente.

### Condiciones de aplicabilidad y calibración

> [PENDIENTE-AUTOR: enumerar de forma compacta las condiciones de
> `ratio_espacio_negativo` y `ratio_nitidez` y explicar de qué
> observaciones y banda vacía sale `VAR_LAPLACIANO_MIN = 50`.]
>
> Un caso que demuestra por qué las dos condiciones de aplicabilidad tienen que ser
> independientes. Una imagen concreta viene de YOLO (abre la primera
> condición de aplicabilidad, cobertura indistinguible de la de un sujeto pequeño legítimo)
> pero su varianza máxima es 27,6, por debajo del corte de textura: se
> cierra por la segunda condición de aplicabilidad y por el motivo correcto --- una escena
> casi sin detalle no permite comparar nitidez, con independencia de si
> el sujeto está bien localizado.
>
> Qué se publica con cada condición de aplicabilidad cerrada. Las dos métricas se publican
> siempre con su valor medido, confianza 0 y fuente_confianza explicando
> qué condición falló --- nunca se silencia una métrica.
> ratio_nitidez.valor_norm = None únicamente en el caso extremo de
> varianza total nula (imagen absolutamente uniforme, un verdadero 0/0);
> con la condición de aplicabilidad cerrada por falta de textura pero varianza no nula, sí hay
> una medición real y valor_norm se calcula con normalidad, porque
> cualquier otra cosa rompería la auditabilidad --- la misma lección que
> dejó angulo_horizonte en el Agente 2.

### Mediciones sobre el corpus de desarrollo

> cv2.grabCut no es determinista: en 27 de las 29 imágenes sobre las que
> se comprobó, dos ejecuciones del mismo proceso sobre la misma entrada
> daban máscaras distintas. La
> causa: GrabCut inicializa sus mezclas de gaussianas con un *k-means*
> interno y cv2.grabCut no expone semilla propia. Con cv2.setRNGSeed()
> se fija la semilla del generador de números pseudoaleatorios interno de
> OpenCV.
>
> El balance de las dos condiciones de aplicabilidad sobre las 40 imágenes del corpus de
> desarrollo es desigual, como corresponde a que una tenga una condición
> y la otra dos: ratio_espacio_negativo queda abierto en 20 imágenes y
> ratio_nitidez en 19. La imagen que separa ambos recuentos es imagen14,
> la única del corpus que cierra por la segunda condición y no por la
> primera: su `bbox` procede de YOLO, pero es un producto blanco sobre
> fondo blanco y la escena no ofrece detalle suficiente para que
> comparar la nitidez de dos regiones signifique algo.

### Alcance y límites de interpretación

> La limitación aceptada procede de los pesos de YOLO: COCO no tiene clases de
> edificio, árbol, montaña ni terreno, por lo que
> la condición de aplicabilidad se cerrará siempre que el sujeto principal sea uno de esos,
> aunque exista y sea perfectamente claro a la vista (un edificio
> aislado, un árbol solitario). Son falsos negativos reales y el sistema
> callará sobre su aislamiento. Se acepta por tres razones:
>
> - La alternativa es el falso positivo simétrico ---tratar como sujeto
> la vegetación o un patrón de muro---, y arreglar un lado estropea el
> otro; se escoge el caso más frecuente, porque en arquitectura y
> paisaje lo habitual es que el interés esté distribuido.
>
> - Tiene precedente propio dentro del mismo proyecto: es la misma
> asimetría con la que se eligió el percentil 99 sobre el 95 en el
> modelo nulo del Agente 2 --- un falso positivo rompe la
> verificabilidad, un falso negativo solo hace callar al sistema, que es
> lo que las condiciones de aplicabilidad existen para hacer.
>
> - Encaja con el propio vector W ---que es coherencia interna del diseño
> y no una confirmación independiente, porque los pesos los fija este
> mismo trabajo---: las clases que YOLO sí cubre son
> justo los contextos donde esta dimensión pesa más (animal 0,32,
> retrato 0,31), y el contexto donde la condición de aplicabilidad se cerrará sistemáticamente
> es arquitectura, con 0,17 ---el peso más bajo que esta dimensión
> alcanza en toda la matriz---. El ajuste no es igual de limpio en paisaje
> (0,22), donde sí se pierde cobertura real; el apartado 9.5 lo declara.

## Agente 4 -- Luz y tono

> El agente 4 mide dos fenómenos sobre la misma imagen convertida a los
> espacios de color CIELAB y HSV: la exposición y distribución tonal, y
> la armonía cromática. Es, junto al agente 2, puramente global, no
> consume `bbox`, ni `centroide_saliencia`, ni `sujeto_discreto`, y su única
> entrada es `ruta_imagen`.
>
> Sus dos métricas presentan una asimetría estructural. La exposición se mide
> siempre porque toda imagen tiene píxeles y un canal de luminancia; la armonía
> cromática requiere suficiente contenido cromático y lleva una condición de
> aplicabilidad propia. Esa
> asimetría gobierna tanto el diseño del motor como la redacción de su
> diagnóstico.

### Qué mide y qué no mide

> **Exposición y distribución tonal.** media_L mide la media del canal L
> de CIELAB, en `[0, 100]`, con `valor_norm = 1 - |media_L - 50| / 50`:
> la adherencia a una exposición centrada en el punto medio de la
> escala, no a un valor de «corrección» fotográfica. `std_L` (dispersión
> del canal L), pct_clipping_sombras y pct_clipping_luces (fracción de
> píxeles con `L < 5` y `L > 95`) se publican como campos planos,
> evidencia que acompaña a la media y no tiene por sí misma una escala
> de adherencia propia, del mismo modo que n_lineas_inliers acompaña a
> score_convergencia en el Agente 2.
>
> **Armonía cromática.** `esquema_cromatico` clasifica la paleta dominante
> en `monocromático`, `análogo`, `complementario` u `otro`
> a partir del matiz de los píxeles cromáticos de la imagen. La etiqueta
> describe la relación entre esos matices, no una valoración de la
> calidad cromática de la fotografía. La regla es la de la rueda
> cromática clásica (apartado 2.3.3), aplicada sobre spread_cromatico
> ---el rango angular que cubren los matices dominantes---: por debajo de
> 30° hay un solo matiz y la paleta es monocromática; por debajo de 60°
> los matices son contiguos y la paleta es análoga; si existe un par de
> matices opuestos, a 180° con una holgura de 30°, es complementaria; y
> `otro` recoge el resto, que es una categoría legítima y no una
> identificación fallida.

### Cálculo, entradas, salidas e implementación

> La implementación se encuentra en
> `src/tfg_multiagente_fotografia/tools/luz_tono_tool.py`. Recibe
> `ruta_imagen`, convierte la imagen a CIELAB y HSV y devuelve las
> métricas tonales, el esquema cromático, los matices dominantes y los
> campos necesarios para auditar su condición de aplicabilidad.
>
> **Exposición y distribución tonal.** Los seis campos graduables de este
> agente llevan escalas de lectura absolutas ---derivadas del dominio de
> la magnitud, no de rangos observados en el corpus de desarrollo--- con
> una única excepción parcial. `media_L` está anclada en `[0, 100]`, con
> el 50 correspondiente al gris medio del 18 %; los porcentajes de
> clipping y la fracción de píxeles cromáticos se citan directamente
> como proporciones del encuadre; y los cortes de spread_cromatico los
> fija la taxonomía cromática clásica enunciada en el apartado 6.5.1,
> sobre la rueda de matiz que presenta el apartado 2.3.3. std_L no tiene una
> convención fotográfica de referencia, pero sí una cota analítica: el
> máximo alcanzable es 50, y solo lo produce el caso degenerado de mitad
> negro puro y mitad blanco puro ---el mismo caso de la prueba analítica---, así que
> se lee como fracción de ese techo. Esta distinción ---entre un umbral
> de aplicabilidad, que sale del corpus y caduca al crecer, y una escala
> de lectura, que sale del dominio matemático de la magnitud y no
> caduca--- es la que separa `RATIO_CROMATICO_MIN` (apartado 6.5.4) de los
> seis campos graduables de exposición y tono.
>
> **Armonía cromática.** El cálculo parte de un agrupamiento por *k-means*
> (k = 5) sobre el matiz de los píxeles cromáticos, tras descartar los
> píxeles de baja saturación o luminancia (`S < 0,15` o `V < 0,15`), que
> corresponden a grises y negros. El agrupamiento no corre sobre el
> ángulo de matiz crudo, sino sobre su proyección al círculo unidad (cos
> θ, sin θ). El centroide se recupera con atan2, que deshace la
> proyección sin necesidad de un caso especial en el cruce por 0°.

### Alternativas descartadas y decisiones de diseño

> **Representación circular del matiz.** En el ángulo crudo, 359° y 1°
> distan 358 unidades, así que *k-means* partiría en dos grupos el rojo
> ---el matiz más frecuente en fotografía--- y el spread saldría enorme
> para una imagen perfectamente monocromática. Por ello, el agrupamiento
> se realiza sobre la proyección al círculo unidad.
>
> **Número fijo de clústeres.** El segundo problema que reveló la
> calibración fue de naturaleza distinta: *k-means* está obligado a
> devolver k grupos exista o no esa cantidad de colores diferenciados,
> de modo que en una foto dominada por un único matiz gasta los cinco
> clústeres partiendo ese mismo matiz en rodajas contiguas. En una
> fotografía con un fondo naranja saturado, tres de los cinco clústeres
> caían todos dentro del naranja, separados por unos pocos grados,
> sumando el grueso de la masa cromática, mientras los dos restantes
> recogían ruido residual. Esto tenía dos consecuencias:
> matices_dominantes publicaba dos o tres matices casi idénticos como si
> fueran colores distintos, y un matiz secundario real quedaba enterrado
> porque sus propios fragmentos se repartían una masa que ninguno
> alcanzaba a superar individualmente. La solución fue una fusión
> aglomerativa de los clústeres antes de aplicar el filtro de masa, con
> centroide calculado como media circular ponderada por masa ---no
> aritmética, porque promediar 350° y 10° de forma aritmética da 180°,
> el color opuesto---.
>
> El peso relativo de cada matiz ---qué fracción de los píxeles
> cromáticos captura cada clúster tras la fusión--- resultó no ser una
> condición de aplicabilidad viable: medido sobre 25 imágenes, varió de 0,281 a
> 0,908 sin ninguna separación clara entre valores altos y bajos. Sí
> resultó necesario, en cambio, como criterio de qué clústeres cuentan como
> matiz dominante: solo se publican en matices_dominantes los que
> superan un peso mínimo, de modo que un clúster sin masa real no se
> reporta como matiz dominante sino que se trata como el artefacto de
> haber pedido cinco grupos a un algoritmo que no puede pedir menos.
>
> Dos decisiones completan el contrato. La primera es que esquema_cromatico no
> lleva valor_norm: sería tentador
> asignárselo según si el esquema resulta identificable, pero esa
> formulación mezclaría aplicabilidad con adherencia, que es exactamente
> la distinción que separa confianza de valor_norm en el contrato común
> del nivel 2 ---si el esquema no es identificable, lo que debe bajar es
> la confianza, no una supuesta adherencia a medias---, y además «otro»
> es una categoría legítima del espacio de resultados, no una
> identificación fallida. La segunda mantiene en cinco los clústeres de *k-means*;
> su fusión y filtrado por masa ya se han descrito en este apartado.

### Condiciones de aplicabilidad y calibración

> **Exposición y distribución tonal.** media_L.confianza vale 1 en todas
> las imágenes porque no existe una condición de aplicabilidad legítima
> para la exposición, toda fotografía tiene un histograma de luminancia,
> y fabricarle una condición de aplicabilidad que pudiera cerrarse introduciría una falsa
> condición de fallo donde no la hay. El precedente dentro del sistema
> es d_equilibrio en el Agente 1, también global y también con confianza
> plena por el mismo argumento.
>
> **Armonía cromática.** La condición de aplicabilidad de esquema_cromatico tiene un solo
> criterio: la fracción de píxeles cromáticos de la imagen tiene que
> superar un umbral mínimo, fijado en 0,10. Por debajo de ese umbral, la
> imagen no ofrece suficiente contenido de color para que un matiz
> dominante signifique algo, y el esquema se cerraría sobre ruido. La
> calibración original de este umbral fue la más holgada del sistema,
> con una separación amplia entre las imágenes que no superaban el corte
> y las que sí; una ampliación posterior del corpus introdujo una
> fotografía que se sitúa apenas por encima del corte, reduciendo ese
> margen sin alterar ninguna clasificación existente. La banda vacía
> observada sobre las 25 imágenes con las que se calibró el umbral iba
> de 0,0002 a 0,2868; sobre el corpus de desarrollo actual, imagen39
> mide 0,1029 y abre la condición de aplicabilidad solo 0,0029 por encima del corte. El umbral
> sigue dentro de la banda, pero desplazado hacia su extremo superior en
> lugar de situarse en su centro, lo que lo deja más expuesto a que una
> imagen futura lo cruce por poco. El apartado 10.3 extrae la lección general de
> esta posición dentro de la banda.
>
> **Fusión de matices.** El corte se calibró en 15°, valor que cae en una
> banda vacía entre las rodajas del mismo matiz (separadas entre sí por
> 3°--8°) y los pares de matices legítimamente distintos (separados por
> 23° o más), y que queda por debajo del corte de 30° que define
> «monocromático», de modo que la fusión nunca junta matices que la
> propia taxonomía considera diferentes.

### Mediciones sobre el corpus de desarrollo

> Fijar la semilla de *k-means* no basta para que el agrupamiento sea
> determinista: con random_state e inicializaciones múltiples fijados,
> 13 de las 25 imágenes sobre las que se comprobó ---algo más de la
> mitad--- devolvían resultados distintos entre dos ejecuciones del
> mismo proceso. La causa no está en la inicialización sino en el orden
> de reducción en coma flotante, que depende del reparto en hilos de las
> bibliotecas de álgebra lineal subyacentes. La discrepancia nunca
> cambiaba la etiqueta final ---del orden de 10⁻¹⁴ grados---, pero
> matices_dominantes es un campo citable, y un campo citable tiene que
> ser reproducible bit a bit por un tercero. La solución fue publicar el
> matiz redondeado a un decimal, lo que además no descarta información
> real: OpenCV ya cuantiza el matiz a 2° al almacenarlo en ocho bits.
> Con esa corrección, el determinismo pasó a ser total: sobre las 40
> imágenes del corpus de desarrollo, dos análisis consecutivos de la
> misma fotografía difieren en 0 casos en el campo publicado, aunque en
> 7 sigan difiriendo en los valores internos que no pasan por el
> redondeo.
>
> La fusión de clústeres, aplicada sobre el reparto de etiquetas del
> corpus, redujo la proporción de imágenes clasificadas como
> «complementario» de 17 de 25 a 8 de 25, es decir, de más de dos
> tercios a poco menos de un tercio. Sin la fusión, cualquier imagen con
> cinco clústeres repartidos sobre un círculo tenía una probabilidad
> geométrica alta de que algún par cayera a 180° ± 30°, con
> independencia de si esa relación cromática existía realmente en la
> fotografía.

### Alcance y límites de interpretación

> **Exposición y distribución tonal.** El valor de referencia conocido de la prueba analítica
> ancla la métrica en tres casos que no dependen de ninguna conversión de
> color: un lienzo negro da media_L = 0,00; uno blanco, 100,00; y un
> lienzo dividido en mitad negra y mitad blanca da media_L = 50,00,
> std_L = 50,00 y un 50 % de clipping en cada extremo. Este último caso
> separa explícitamente adherencia de calidad: valor_norm = 1,00 en una
> imagen que no es fotográficamente «correcta» en ningún sentido, porque
> valor_norm mide únicamente cercanía al centro de la escala de
> luminancia, no si esa exposición sirve a la fotografía.
>
> **Armonía cromática.** La distinción entre una fotografía en blanco y
> negro y una fotografía «monocromática» no depende del número de
> clústeres que produce *k-means*, sino precisamente de esta condición
> de aplicabilidad: una imagen sin apenas saturación tiene el esquema cromático
> cerrado ---no hay paleta que describir---, mientras que una imagen
> dominada por un único matiz saturado tiene el esquema abierto con la
> etiqueta «monocromático». Confundir ambos casos convertiría la
> etiqueta en una descripción ambigua de dos fenómenos distintos.
>
> El propio filtro de saturación y luminancia que hace posible el
> análisis de matiz introduce una limitación estructural: la métrica
> describe únicamente el matiz de los píxeles cromáticos, y todo lo
> acromático ---blancos, negros y grises--- es invisible para ella por
> construcción. En una fotografía cuyo sujeto es acromático sobre un
> fondo saturado, ese sujeto no existe para esquema_cromatico, y la
> etiqueta «monocromático» describe solo el fondo, perdiendo por
> completo la relación de contraste entre sujeto y fondo que
> compositivamente define la imagen. Esto no es un defecto del algoritmo
> sino un límite de lo que la regla puede afirmar, y se compensa sin
> modificar el flujo de procesamiento de cómputo, en dos frentes. Primero, la fracción
> de píxeles cromáticos deja de ser solo la criterio de aplicabilidad y se
> convierte en un hecho compositivo citable por derecho propio: su
> complementario es la fracción acromática del encuadre, y el
> diagnóstico debe leerla y citarla con independencia de si el esquema
> cromático está abierto o cerrado. Segundo, la semántica de la etiqueta
> se acota explícitamente: «monocromático» significa un único matiz
> dominante entre los píxeles cromáticos de la imagen, no que la imagen
> tenga un único color.
>
> La asimetría entre las dos métricas de este agente tuvo además una
> consecuencia inesperada en la redacción del diagnóstico: pedirle al
> modelo que la expresara lo llevó a narrar el mecanismo del sistema en
> lugar de la fotografía. Es el defecto que se detectó por primera vez
> aquí y se confirmó después en el Agente 3, y se cuenta con su causa y
> su corrección en el apartado 6.6.

## Del informe al diagnóstico: diseño de plantillas de instrucciones y control de la redacción

> `agents.yaml` y `tasks.yaml` suman 1612 líneas, a las que se añaden cerca
> de 1600 líneas de description en los esquemas Pydantic dirigidas al modelo de
> lenguaje. Por tanto, buena parte de lo que el sistema dice sobre una fotografía
> no está en NumPy ni en OpenCV, sino en las plantillas de instrucciones. Estas se
> sometieron al mismo ciclo de calibración que las condiciones de aplicabilidad
> ---contrato inicial, validación, revisión y corrección---, aunque aquí se ajusta
> lenguaje natural y no un umbral numérico.
>
> Para redactar todas las plantillas de instrucciones se tomó como referencia el libro de
> Chip Huyen, «Ingeniería de IA», que ofrece una guía para elaborar
> plantillas de instrucciones y reúne buenas prácticas de diseño. Estas
> prácticas abarcan campos como escribir instrucciones claras y
> explícitas (sin ambigüedades, rol específico, adjuntar ejemplos,
> especificar la entrada y la salida), aportar suficiente contexto, dividir
> tareas, hacer iteraciones de plantillas de instrucciones, etc. Son la base para establecer
> un contrato de redacción tanto en el orquestador y el crítico, como en los
> especialistas.
>
> El contrato de redacción es idéntico para los cuatro especialistas y
> queda enunciado en el apartado 4.5.3, junto con las cinco cosas que el
> diagnóstico tiene que aportar para que el paso por el modelo de
> lenguaje no sea un coste sin retorno (apartado 4.5.2). Lo que sigue no
> repite ese contrato, sino que cuenta lo que costó hacerlo cumplir: qué
> falló, por qué, y qué forma tuvo que tomar cada regla para que el
> modelo la respetara.
>
> Validar una plantilla de instrucciones contra un único caso resultó
> insuficiente. Tras una actualización de la
> versión de CrewAI que no modificó ni una cifra de los informes ---los
> cinco especialistas devolvieron el mismo JSON, campo a campo, en ambas
> versiones---, tres de los cuatro diagnósticos incumplieron reglas que
> habían respetado en su validación original. El Agente 4 reintrodujo,
> de forma casi literal, tres defectos que ya se habían corregido en su
> momento; el Agente 1 empezó a introducir muletillas de relleno; y el
> Agente 2 cometió el fallo más grave que contempla el contrato del
> sistema: citó el valor numérico de una métrica con confianza cero. El
> único especialista que salió indemne de las dos validaciones fue el
> Agente 3, y no por casualidad: se había escrito más tarde, cuando ya
> se conocían los tres modos de fallo detectados en los otros tres, y
> los prohibía explícitamente uno por uno, mientras que cada uno de los
> demás solo prohibía los defectos que había mostrado en su propia
> validación. La explicación más consistente no es la
> actualización del *framework* ---que no movió ni una cifra de los
> informes, y de la que el Agente 3 salió indemne---, sino la
> variabilidad del modelo de lenguaje sobre plantillas de instrucciones frágiles, sumada al
> hecho de que cada agente antiguo solo tenía una ejecución validada.
>
> La corrección sustituyó la práctica seguida hasta entonces: añadir un párrafo de
> prohibición al agente afectado cada vez que aparecía un defecto nuevo
> ---una estrategia que produce instrucciones cada vez más largas, en
> las que las reglas importantes se diluyen entre las accesorias y cada
> parche queda sobreajustado al caso concreto que lo motivó---. En su
> lugar, se incorporó un bloque idéntico de diez líneas en los cuatro
> especialistas: una comprobación final de cinco criterios con los que
> el modelo relee su propio borrador y borra las frases que los
> incumplan ---que no hable de la fotografía, que no incluya un número
> ausente del informe, que no se apoye en una métrica con confianza
> cero, que no emita un juicio de valor, que no repita algo ya dicho---.
> Dos propiedades explican el resultado. Cambia el modo de la instrucción, de una
> lista de negaciones que hay que tener presentes mientras se redacta a
> un paso de revisión explícito aplicable frase a frase al final; y le
> concede al modelo permiso explícito para acortar el texto, atacando
> directamente el sesgo que había producido los tres regresos ---la
> tendencia a alargar el diagnóstico para parecer más completo---. Sobre
> la misma imagen ---una sola--- con la que se había detectado el
> problema, los cuatro
> diagnósticos volvieron a estar limpios en las cinco familias de
> defecto que se venían vigilando, y los textos se acortaron
> precisamente donde antes sobraban.
>
> El fallo más grave ---citar el valor de una métrica cerrada--- exigió,
> sin embargo, una intervención más profunda que una comprobación final,
> porque su causa no era un descuido de redacción sino una contradicción
> interna de la propia plantilla de instrucciones: el `backstory` prohibía apoyarse en una
> métrica sin confianza, y al mismo tiempo el `expected_output` pedía
> explícitamente los puntos del diagnóstico que citaban esa misma
> métrica, sin condicionar esa petición al valor de su confianza. Ante
> esa contradicción, el modelo obedecía sistemáticamente la instrucción
> concreta ---«escribe esto»--- por encima de la prohibición general
> ---«no hagas aquello»---. Este patrón se repitió, con variaciones
> menores, en los cuatro especialistas y también en el crítico, y la
> corrección fue siempre la misma: no añadir una prohibición más, sino
> eliminar la instrucción que contradecía a la prohibición existente. El
> `expected_output` de cada especialista pasó a bifurcarse explícitamente
> al principio según la confianza de la métrica que gobierna su condición de aplicabilidad:
> con confianza cero, los puntos que citarían esa métrica sencillamente
> no se escriben, y la plantilla de instrucciones declara de forma expresa que un
> diagnóstico más corto es el resultado correcto en ese caso, no una
> carencia que compensar. El efecto se midió directamente en la longitud
> del texto: con ambas condiciones de aplicabilidad de un especialista cerradas, el diagnóstico
> se redujo a una o dos frases de menos de treinta palabras, frente a
> las cuatro o cinco frases que antes rellenaban el hueco citando
> precisamente la métrica que debían callar.
>
> Otros defectos recurrentes se corrigieron con el mismo principio de sustracción.
> La instrucción de redondeo, que originalmente autorizaba
> «redondear a cuatro decimales», producía truncamientos en lugar de
> redondeos correctos, porque el modelo interpretaba la instrucción como
> un límite de longitud y no como una operación aritmética; la
> reformulación explicita que lo que se escribe tiene que ser el
> redondeo correcto a esa precisión y no un corte de cifras, con
> ejemplos resueltos, y añade la salida de que, ante la duda, es
> preferible escribir menos decimales que arriesgar una cifra
> incorrecta. La costumbre de citar el campo fuente_confianza copiando
> la frase completa dentro de los corchetes de cita se prohibió
> explícitamente, tanto porque viola la regla de citar solo un escalar
> como porque dos métricas de un mismo informe pueden compartir ese
> nombre de campo, y la cita quedaría ambigua sobre a cuál se refiere.
> Trasladar al texto los números de una escala de lectura ---una
> constante de referencia que no es un campo del informe, como el máximo
> teórico de una magnitud--- se prohibió por el mismo motivo que se
> prohíbe inventar cifras: obliga a auditar un número que no aparece en
> ningún informe; la única excepción admitida es cuando el número
> trasladado es una transformación exacta y recomputable del valor
> citado, como el factor multiplicativo que deriva de un índice
> normalizado, porque en ese caso cualquiera puede reconstruir la cifra
> desde la propia cita.
>
> Una última clase de defecto apareció en el Agente 4 y se confirmó en el Agente
> 3: en lugar de describir la fotografía, el modelo describía el propio sistema
> ---condicionales
> hipotéticos sobre qué habría escrito si una confianza hubiera sido
> distinta, explicaciones de por qué una métrica no tiene condición de aplicabilidad, frases
> que calificaban su propia afirmación como «un hallazgo legítimo y no
> un fallo del análisis»---. La causa era la misma: el
> propio `expected_output` pedía explicar la asimetría de fiabilidad entre
> las métricas del agente, o calificar de antemano una conclusión
> posible, y el modelo terminaba citando casi literalmente esa
> instrucción como si fuera parte del diagnóstico. La corrección
> consistió en reformular esas instrucciones para que la asimetría del
> agente gobierne exclusivamente lo que el especialista afirma ---con
> confianza plena, cualquier mención de la asimetría se omite--- y no lo
> que explica sobre su propio diseño.

# Nivel 3 -- Crítico, síntesis e interfaz

El nivel 3 cierra la jerarquía del sistema. Recibe las cuatro salidas
del nivel 2, el informe de métricas de cada especialista, en su dominio
natural y normalizado, junto con el diagnóstico en prosa que cada
especialista ya redactó a partir de esas métricas, más el vector de
pesos contextuales W y la etiqueta de contexto que fijó el orquestador.
Su tarea no es volver a verbalizar cifras crudas, sino arbitrar y sintetizar los
cuatro diagnósticos mediante dos criterios independientes. W fija el orden del
discurso al ordenar las dimensiones según el peso del contexto; la confianza de
cada métrica decide qué puede citarse dentro de ellas. Un peso
alto no convierte en citable una métrica cerrada, y una cobertura baja no
degrada la posición de su dimensión ---el apartado 7.1.4 argumenta por qué---.
Sobre esa base resuelve además las tensiones que puedan surgir
entre dimensiones que, por diseño, se midieron de forma independiente y
ortogonal entre sí.

El anteproyecto planteaba este nivel con dos agentes, uno métrico y uno
crítico: el primero interpretaría los resultados en parámetros técnicos
cuantificables y el segundo haría la síntesis cualitativa. Los dos se
han fusionado en uno solo porque la parte cuantificable no necesitaba un agente: ya
la resuelven dos piezas que no son modelos de lenguaje: el informe que
cada especialista entrega, calculado con NumPy y OpenCV, y el arbitraje
determinista que se describe en el apartado 7.1. Un agente métrico que
volviera a interpretar esos números en una segunda pasada habría
devuelto autoría generativa justo al sitio del que se estaba retirando.

Al diseñar este agente, se detectó que el crítico sería
el único componente del sistema cuya salida fuera 100 % autoría del
modelo de lenguaje. Esto debilitaría la verificabilidad, por lo que se
le da al crítico una parte determinista, calculada sin ningún modelo de
lenguaje, sobre la que después se apoya la parte generativa: el arbitraje del
apartado 7.1. Es también
lo que evita añadir un coste de ejecución específico: la función del agente
métrico no desaparece, sino que se conserva en una pieza recomputable.

El resto del nivel conserva esa separación de autorías. Comprende la síntesis en
prosa que cita el arbitraje y los diagnósticos (apartado 7.2); las tensiones entre
dimensiones (7.3); la imagen como tercera fuente, ciega a las métricas (7.4); el
alcance del juicio del crítico (7.5); y la interfaz que entrega el resultado con
su evidencia visual (7.6).

## Arbitraje determinista: orden por W, métricas citables y cobertura por dimensión

### Qué resuelve

> El arbitraje es el mecanismo que aplica el vector W sin que ningún
> modelo de lenguaje intervenga en el cálculo. Antes de que el crítico
> escriba una sola frase, `PrioridadTool`, que envuelve al motor
> `PrioridadEngine`, lee los cuatro informes de métricas ya escritos en
> disco, consulta la fila de W correspondiente al contexto de la imagen
> y produce una estructura tipada, `ArbitrajeCritico`, que fija tres
> cosas: en qué orden se debe hablar de las cuatro dimensiones, qué
> métricas concretas son citables y qué fracción de cada dimensión quedó
> realmente medida. El LLM recibe ese arbitraje como un dato más de su
> contexto, igual que recibe los diagnósticos, no lo genera, lo consume.

### Cómo se construye

> `PrioridadEngine.arbitrar(ruta_imagen, etiqueta_contexto)` no recibe los
> informes por parámetro desde la plantilla de instrucciones: los lee de los cuatro ficheros
> `salida_*.json` que el nivel 2 ya dejó escritos en
> outputs/ejecucion/{stem}/. Así se evita que el LLM retranscriba los 70 campos
> de los cuatro informes. El motor incorpora una guarda explícita: antes de
> construir el arbitraje comprueba que los cuatro informes corresponden
> a la imagen que se está analizando, deduciéndolo de su
> `verificacion_path` con coincidencia por el nombre completo del fichero
> (de modo que, por ejemplo, imagen3 no case por error de subcadena con
> imagen30). Si los informes no coinciden, el motor falla de forma
> ruidosa (`ValueError`) en lugar de arbitrar en silencio sobre datos de
> otra fotografía.
>
> Las métricas se detectan por su forma, no por una lista de nombres
> cerrada: cualquier campo del informe que tenga la estructura
> `{valor, confianza, ...}` se trata como una métrica candidata a cita. Por ello,
> añadir o renombrar una métrica no obliga a modificar el arbitraje ni puede
> excluirla en silencio del cómputo de cobertura. Los campos planos del
> informe (recuentos, listas, varianzas crudas) se excluyen
> deliberadamente de este criterio: no son afirmaciones envueltas en
> confianza, y contarlos inflaría el denominador de la cobertura con
> evidencia que no es, en sí misma, citable de forma independiente.

### La estructura de salida: ArbitrajeCritico

> El arbitraje se emite con seis campos:
>
> - `etiqueta_contexto`: la clase que el clasificador asignó a la
> fotografía, que es la clave con la que se selecciona la fila de W.
>
> - `vector_pesos`: la fila de W efectivamente aplicada, tal y como la
> devuelve `VectorPesosTool` para la etiqueta de contexto detectada.
>
> - `orden_dimensiones`: las cuatro dimensiones compositivas ordenadas
> por el peso que W les asigna en ese contexto, de mayor a menor. Es la
> traza auditable de que W se aplicó de verdad y no es un adorno de la
> plantilla de instrucciones.
>
> - `metricas_citables`: la lista de rutas dimensión.métrica cuya
> confianza es mayor que 0, es decir, lo que el crítico puede afirmar.
>
> - `metricas_cerradas`: la lista complementaria, con las rutas cuya
> confianza vale 0. Junto con la anterior marca la frontera exacta entre
> lo que se puede citar y lo que está prohibido citar.
>
> - `cobertura_por_dimension`: la fracción de métricas citables dentro
> de cada dimensión. Es un recuento de aplicabilidad, no una agregación
> de calidad: dice cuánto de esa dimensión pudo medirse, no si lo medido
> es bueno o malo.

### Dos decisiones de diseño

> El orden de las dimensiones se determina únicamente a partir de su
> peso W, sin modificarlo en función de la cobertura obtenida. Esta
> decisión permite distinguir entre la importancia de una dimensión y la
> capacidad del sistema para medirla. Si una dimensión tiene un peso
> elevado pero cobertura cero, reducir su peso ocultaría precisamente
> una limitación relevante: el sistema considera importante ese aspecto
> de la imagen, pero no dispone de información suficiente para
> evaluarlo.
>
> Este comportamiento se observó, por ejemplo, en una fotografía de
> paisaje donde la dimensión espacio y aislamiento del sujeto tenía el
> tercer mayor peso, pero cobertura nula al no disponer de una detección
> fiable del sujeto. En lugar de desplazar esta dimensión o repartir su
> peso entre las restantes, el sistema mantiene su posición y hace
> explícita la ausencia de información. De esta forma, una métrica
> cerrada no se interpreta erróneamente como una dimensión poco
> relevante.
>
> Cuando varias dimensiones tienen el mismo peso, se utiliza un orden
> canónico fijo como criterio de desempate. Esto garantiza que una misma
> entrada produzca siempre la misma jerarquía. En cualquier caso, estos
> pesos se emplean únicamente para ordenar y priorizar el contenido de
> la crítica, y no para combinar las métricas en una puntuación global
> de calidad.

## Síntesis con LLM: cierres, formato de cita y estructura de salida por autoría

### Qué recibe el crítico y qué produce

> `critica_task` agrega las cinco tareas anteriores: recibe en su contexto los cuatro pares
> `{informe, diagnostico}` de los especialistas y la etiqueta de contexto
> del orquestador. La plantilla impone tres pasos en un orden que no permite
> invertir. Primero invoca la herramienta Arbitraje de prioridades (la
> envoltura de PrioridadEngine, descrita en el apartado 7.1) y trata su resultado no
> como una sugerencia sino como la estructura misma de la crítica. Después, si
> se dispone de ella (`vio_imagen = True` o `False`), invoca
> `LecturaVisual` y descarta de inmediato cualquier observación que toque
> las cuatro dimensiones ya medidas; y solo entonces redacta recorriendo
> las dimensiones en el orden que fijó el arbitraje y tomando de cada
> especialista su `diagnostico` ya redactado, sin volver a interpretar las
> métricas desde cero ni transcribir el informe entero.
>
> El resultado es `CriticaCompositiva`, una estructura de siete campos
> que, como en el nivel 2, separa la evidencia por quién la produjo, tal
> como recoge la Tabla 7.1:

**Tabla 7.1.** Los siete campos de `CriticaCompositiva`, con su
autoría y su contenido.

  -----------------------------------------------------------------
  Campo                    Autoría              Contenido
  ------------------------ -------------------- -------------------
  arbitraje                Determinista         Transcrito sin
                           (PrioridadTool)      alterar: sin
                                                reordenar, sin
                                                recalcular, sin
                                                omitir ni añadir
                                                campos

  afirmaciones             LLM, con cita        Una o más por
                           obligatoria          dimensión con
                                                métricas citables,
                                                en el orden del
                                                arbitraje

  tensiones                LLM, con cita        Hallazgos
                           obligatoria          contradictorios
                                                entre dimensiones
                                                resueltos con W

  observaciones_visuales   LLM, sin cita        Contenido que
                           posible              ninguna métrica
                                                mide; vacío si no
                                                hubo lectura visual

  salvedades               LLM, ancladas al     Una por cada
                           arbitraje            métrica de
                                                métricas_cerradas

  sintesis                 LLM                  El único campo que
                                                lee el fotógrafo

  vio_imagen               Indicador registrado Capacidad de ver la
                                                imagen
  -----------------------------------------------------------------

> Esta partición permite comprobar, campo a campo y sin otro modelo de lenguaje,
> que ninguna cifra de sintesis aparece sin pasar antes por afirmaciones o
> tensiones, y que toda frase sin cifra corresponde a una observación visual o a
> una salvedad.

### Los cuatro cierres

> El crítico tiene su equivalente de las condiciones de aplicabilidad del nivel
> 2: cuatro reglas que delimitan qué puede hacer con sus tres fuentes.
>
> - **Sin cita no hay afirmación métrica.** Toda afirmación sobre las
> cuatro dimensiones medidas se apoya en una métrica de
> `metricas_citables`; lo demás se marca como observación visual o se omite. La
> regla se extiende a los
> calificativos («un contraste notable», «tonos cian-azulados»), que son
> afirmaciones encubiertas y necesitan su cifra detrás. Para poder
> graduarlos sin inventar, la plantilla de instrucciones habilita explícitamente los campos
> planos de los informes (`std_L`, `matices_dominantes`, `margen_patron`, los
> recuentos y varianzas) como evidencia citable de pleno derecho aunque
> no figuren en `metricas_citables` ---esa lista solo recoge las métricas
> con condición de aplicabilidad---, con la única condición de que pertenezcan a una dimensión
> abierta.
>
> - **No reabrir una condición de aplicabilidad cerrada.** Con una métrica en `metricas_cerradas` no
> hay número, ni siquiera «aproximado» o «a ojo». Se puede decir que esa
> propiedad no es medible en esa escena y describirla cualitativamente;
> nunca cuánto vale.
>
> - **No alterar un valor.** Ni recalcular, ni combinar dos métricas en una
> tercera, ni redondear más allá de los cuatro decimales que autoriza el
> sistema.
>
> - **Jurisdicción.** Una observación visual no puede versar sobre ninguna
> de las cuatro dimensiones medidas, aunque al crítico le parezca que
> las contradice: lo que se ve no matiza, no rebaja y no corrige una
> medición. La regla operativa que lo resume, «identificar no es medir»,
> tiene su propio apartado (7.4.4) porque es la que hace compatible el
> canal visual con el resto del sistema.

### Formato de cita y disciplina de redacción

> Toda cifra va entre corchetes, como un escalar y con notación de punto
> (`[d_tercios.valor = 0.0648]`), nunca como diccionario ni fragmento de
> JSON; campos_citados recoge la ruta completa de cada cita y tiene que
> coincidir exactamente con lo citado en el texto. La regla de redondeo
> es la misma que rige en el nivel 2 y se formula de forma explícita
> para evitar el truncamiento: «lo que escribas tiene que ser el
> redondeo correcto a esa precisión» ---0.5939780... se escribe 0.5940,
> nunca 0.5939--- con la salida de que, ante la duda, es preferible
> escribir menos decimales bien redondeados que cuatro mal cortados. Los
> pesos de W solo se citan como cifra cuando sostienen una decisión de
> jerarquía; el resto del arbitraje no se narra en prosa, porque ya
> viaja íntegro en el campo arbitraje y repetirlo sería ruido.
>
> Cuando el crítico dispone de la lectura visual, la cita convive además
> con el sustantivo que esa lectura aporta, bajo un reparto estricto que
> desarrolla el apartado 7.4.4: el sustantivo lo pone siempre la lectura
> visual y el número siempre la métrica.
>
> [PENDIENTE-AUTOR: falta una figura con un ejemplo real de crítica final ---un
> fragmento de `sintesis` de una ejecución ya archivada en `outputs/ejecucion/` o
> `outputs/validacion_2026-08-21/`, con las citas `[métrica.campo = valor]`
> resaltadas y una llamada lateral a la afirmación del `informe` del que sale cada
> una--- para hacer visible de un vistazo la premisa de verificabilidad que sostiene
> el TFG, en vez de solo describir el formato de cita en prosa.]

### Qué puede concluir el crítico

> El único juicio que se permite es si la fotografía se acoge a las
> convenciones que más pesan en su contexto, y solo bajo dos condiciones:
> que se apoye en una afirmación citada y que nombre el contexto que la
> sostiene. Quedan prohibidos el gusto personal, el juicio sobre el autor
> y cualquier conclusión sobre una dimensión con métricas cerradas. El
> apartado 7.5 desarrolla esa frontera, con el ejemplo que la fija en la
> propia plantilla de instrucciones y las alternativas que se sopesaron al decidirla.

### Comprobación final

> Antes de entregar, la plantilla exige releer cada frase y borrarla si incurre
> en alguno de seis defectos: (1) hablar del sistema en vez
> de la fotografía ---la clase de fallo más reincidente, con la regla
> «la escena nunca carece de nada: es como es»---; (2) contener una
> cifra ausente de los informes o de W; (3) apoyarse en una métrica
> cerrada; (4) emitir un juicio de gusto o sobre el autor; (5) repetir
> algo ya dicho; (6) que una observación visual toque encuadre, líneas,
> espacio, nitidez, luz o color ---con la salvedad explícita de que
> nombrar el objeto que una métrica mide no cuenta como invasión---. Es
> el mismo mecanismo que corrigió tres defectos de prosa en los cuatro
> especialistas: seis criterios de borrado, no de prohibición añadida,
> porque el sesgo observado del modelo es alargar el texto para parecer
> completo. El mecanismo prioriza así una crítica breve frente a una más extensa.
>
> Durante la validación se comprobó, sobre fragmentos aislados de la
> propia plantilla de instrucciones, que sin la herramienta de arbitraje disponible el
> modelo inventa una fila de W plausible pero incorrecta en lugar de
> negarse a responder ---una ejecución dio 0,20/0,35/0,15/0,30 para
> paisaje cuando la fila vigente era otra---. Es la evidencia más
> directa de por qué el arbitraje tiene que llegar como salida de una
> herramienta verificada y no como algo que el LLM reconstruye de
> memoria, y una de las razones por las que el modelo del crítico se
> fijó en gemini-2.5-pro (apartado 4.7.2). La estructura por
> autoría no solo permite auditar la salida: es lo que hace visible, y
> por tanto corregible, que la parte generativa se apoye realmente en la
> parte determinista y no la sustituya por una aproximación verosímil.

## Tensiones entre dimensiones: el bloque que justifica la arquitectura multiagente

### Qué es una tensión y por qué es importante

> Una Tension relaciona dos o más propiedades de la fotografía, medidas por
> especialistas que no se comunican entre sí, que apuntan en sentidos distintos.
> El crítico las resuelve citando W. El esquema tipa el hallazgo en cuatro campos (texto, dimensiones,
> resolucion_por_w y campos_citados).
>
> El argumento es estructural, no retórico. Los cuatro especialistas del
> nivel 2 se diseñaron deliberadamente ortogonales y sin contexto
> cruzado: cada tarea de CrewAI declara de forma explícita qué recibe, y
> ningún especialista ve el informe de otro. Esa incomunicación es lo
> que garantiza que cada dimensión se mida sin contaminarse por las
> demás, pero tiene un coste: ningún componente del sistema, antes del
> crítico, puede decir nada sobre la relación entre dos dimensiones. El
> único punto del sistema donde converge la información suficiente para
> detectar que, por ejemplo, la estructura lineal de una escena empuja
> la mirada hacia un lugar donde no hay ningún sujeto que aislar, es la
> síntesis del nivel 3. Es también el único punto donde eso puede
> decirse con una cita a cada lado en vez de como una impresión: un LLM
> generalista mirando la foto podría escribir una frase parecida, pero
> sin dos métricas verificables detrás sería una afirmación de gusto, no
> un hallazgo trazable.

### Condiciones que impone el esquema

> dimensiones exige un mínimo de dos elementos por definición: con una
> sola no hay tensión posible, solo una afirmación. campos_citados exige
> al menos una cita, lo que hereda de forma transitiva el primer cierre
> del apartado 7.2: una tensión solo puede construirse sobre evidencia
> de metricas_citables, nunca sobre una métrica cerrada. En consecuencia, una
> tensión nunca puede rellenar un hueco de cobertura. Si una de las dos
> dimensiones implicadas no tiene ninguna métrica abierta, no hay
> evidencia con la que construir su lado de la tensión, y esa dimensión
> sin cobertura se trata como corresponde ---con una salvedad---, no
> disfrazada de conflicto con otra.
>
> resolucion_por_w es, además, el único lugar del sistema ---fuera del
> propio arbitraje--- donde está permitido hablar de importancia y
> jerarquía. Los cuatro especialistas lo tienen vetado por diseño
> precisamente porque no conocen W; el crítico es el único nivel que sí,
> y este campo es donde ese conocimiento se hace visible con una cita
> concreta al peso y a la etiqueta de contexto, no como una opinión sin
> anclar.

### Tensión como hallazgo, no cualquier discrepancia

> El bloque puede ir vacío, y la plantilla de instrucciones lo dice de forma explícita: no
> toda fotografía presenta una tensión real, e inventar una para
> rellenar el apartado sería peor que omitirlo. Es el mismo criterio de
> contención que gobierna el resto del sistema ---preferir el silencio a
> la falsa precisión--- aplicado a la síntesis.
>
> Una tensión legítima debe distinguirse de un defecto de diseño. En una fase
> anterior del proyecto, sobre una
> misma fotografía y el mismo cuadro delimitador, el Agente 1 llegó a afirmar
> dónde estaba el sujeto con confianza plena mientras el Agente 3
> declaraba, sobre esa misma evidencia, que la escena no ofrecía un
> objeto que aislar. Eso no era una tensión compositiva: era una
> contradicción interna causada porque dos especialistas aplicaban
> criterios de aplicabilidad distintos al mismo dato de entrada. La
> respuesta correcta no fue dejar que el crítico la narrara como un
> hallazgo, sino corregirla aguas arriba, en el nivel 2, unificando la
> condición de aplicabilidad del Agente 1 al mismo criterio (`fuente == "yolo"`) que ya usaba
> el Agente 3. Una tensión genuina parte de dos mediciones coherentes
> cada una consigo misma que, al leerse juntas, cuentan algo que ninguna
> de las dos cuenta por separado; una contradicción entre especialistas
> sobre el mismo hecho es un error de calibración y se resuelve en el
> nivel donde se mide, no en el nivel donde se narra.

### Estado de validación

> El mecanismo de tensiones se ha ejercitado tanto con el crítico ciego
> a la imagen ---la rama sin lectura visual, en la que se validó por
> primera vez--- como en ejecuciones posteriores
> con la tercera fuente activa ---la rama con lectura visual---. Esas son las
> dos configuraciones en las que el crítico puede operar, las gobierna una sola
> constante y el apartado 8.4.4 las describe con detalle. De las 27
> ejecuciones completas archivadas en outputs/ejecucion/ que conservan su
> crítica, 17 contienen una tensión emitida y 10 ninguna; en ningún caso
> se emitió más de una.
>
> La frecuencia con la que aparecen estas tensiones no se interpreta,
> sin embargo, como una propiedad estable del sistema, ya que depende
> necesariamente de las características de las imágenes analizadas y de
> que existan al menos dos dimensiones con evidencia citable cuya
> relación permita formular un conflicto compositivo. Por este motivo,
> un porcentaje de ejecuciones con tensión solo resulta descriptivo
> respecto al corpus concreto sobre el que se calcule y no constituye,
> por sí mismo, una medida de calidad del mecanismo.
>
> La validación relevante se sitúa en otro nivel: comprobar que una
> tensión solo aparece cuando existen evidencias citables suficientes,
> que relaciona al menos dos dimensiones distintas y que su resolución
> respeta los pesos contextuales (W) definidos antes de la intervención
> del crítico.
>
> Lo que queda establecido en este apartado es el mecanismo: la tensión
> está sometida a las mismas reglas de cierre que el resto de la crítica
> y depende de la capa determinista descrita en el apartado 7.1. Incluso
> en el bloque de carácter más interpretativo, el LLM no puede
> introducir libremente la evidencia sobre la que razona; únicamente
> puede relacionar y jerarquizar información que una herramienta no
> basada en LLM ha declarado previamente citable.

## La tercera fuente: lectura visual ciega y jurisdicción del canal visual

### Por qué se incorpora una tercera fuente

> El crítico construye su salida a partir de tres fuentes con funciones
> distintas. La primera está formada por los informes y diagnósticos de
> los cuatro especialistas y aporta la evidencia sobre las dimensiones
> medidas. La segunda es el arbitraje compuesto por `W` y la
> etiqueta de contexto, que fija el orden y el énfasis del discurso. La
> tercera es la propia fotografía, procesada mediante LecturaVisualTool.
>
> La incorporación de la imagen responde a una limitación concreta de
> los informes métricos. Los especialistas pueden determinar dónde se
> encuentra un sujeto, cuánto espacio ocupa, cómo se distribuye la
> saliencia o qué estructura tonal presenta la escena, pero no
> necesariamente saben qué representan esos elementos. Una coordenada
> puede indicar la posición de una figura sin revelar que se trata de
> una persona de espaldas al mar; un área etiquetada como fondo no
> explica si contiene un acantilado, una multitud o una pared; y una
> detección fallida no implica que la fotografía carezca de un
> protagonista reconocible para un observador.
>
> Sin esta tercera fuente, la crítica puede ser correcta en términos
> métricos y, al mismo tiempo, resultar excesivamente abstracta. La
> prosa queda limitada a expresiones como «el sujeto», «la figura» o «la
> escena», incluso cuando la identidad y las relaciones entre los
> elementos son esenciales para que el fotógrafo comprenda el análisis.
> La lectura visual se incorpora para cubrir ese hueco semántico, no
> para repetir las mediciones existentes.

### Qué significa que la lectura sea ciega

> Se refiere a la herramienta que observa la imagen, no al crítico
> final. LecturaVisualTool recibe únicamente ruta_imagen. No conoce la
> etiqueta de contexto, el vector `W`, los informes métricos ni los
> diagnósticos de los especialistas. Su modelo de entrada impide pasarle
> cualquier otro dato.
>
> Esta separación evita una fuente de contaminación difícil de detectar.
> Si el modelo visual recibiera previamente que «el sujeto está
> descentrado» o que «la escena presenta una perspectiva marcada»,
> podría producir una descripción concordante porque ya conoce la
> respuesta esperada. La aparente coincidencia entre fuentes no
> aportaría evidencia independiente: sería la repetición generativa de
> una conclusión introducida en el contexto. Una lectura informada por
> las métricas actuaría como un eco; una lectura ciega conserva la
> posibilidad de aportar información nueva.
>
> La ceguera también sostiene la comparación experimental entre el
> crítico con imagen y el crítico sin imagen. Si el canal visual tuviera
> acceso a los informes, la rama visual no mediría el efecto de observar
> la fotografía, sino el de reformular por segunda vez los mismos datos.
> El aislamiento de la entrada permite atribuir las diferencias entre
> ambas ramas a la incorporación de contenido visual.
>
> La llamada se realiza directamente a gemini-2.5-flash desde la
> herramienta, con temperatura 0,2. Antes del envío, la imagen se reduce
> de forma proporcional hasta un máximo de 1024 píxeles en su lado mayor
> y se recodifica como JPEG. Esta resolución resulta suficiente para
> reconocer contenido general y evita enviar fotografías de muy alta
> resolución cuando el objetivo no es medir detalle fino. La herramienta
> solicita entre tres y seis observaciones breves y devuelve únicamente
> texto estructurado.

### Jurisdicción propia del canal visual

> La independencia de la lectura no implica que pueda pronunciarse sobre
> cualquier aspecto de la imagen. Cada fuente tiene una jurisdicción
> específica. Los informes del nivel 2 son la única fuente autorizada
> para sostener afirmaciones sobre las cuatro dimensiones medidas:
> composición espacial; líneas y dirección; espacio y aislamiento del sujeto; y
> luz y tono. `W` solo puede establecer importancia y
> orden. El canal visual queda limitado a aquello que ninguna métrica
> representa.
>
> Dentro de esa jurisdicción, la lectura puede identificar:
>
> - qué representa la fotografía y dónde tiene lugar la escena;
>
> - quién o qué actúa como sujeto principal;
>
> - qué está haciendo y hacia dónde se orienta;
>
> - qué elementos lo acompañan o compiten con él;
>
> - qué contienen realmente el primer plano, el fondo o las zonas
> aparentemente vacías;
>
> - y qué relaciones visibles existen entre los objetos, como
> superposición, dirección de una mirada o disposición delante-detrás.
>
> En cambio, tiene prohibido juzgar el centrado, los tercios, el
> equilibrio, la perspectiva, los puntos de fuga, el horizonte, el
> espacio negativo, la nitidez, el aislamiento, la exposición, el
> contraste o el esquema cromático. Tampoco puede valorar si esos
> elementos están bien resueltos ni proponer mejoras.
>
> La restricción se aplica dos veces: la plantilla de `LecturaVisualTool` impide
> generar observaciones sobre las dimensiones medidas y `critica_task` ordena al
> crítico descartar
> cualquier observación recibida que invada esos ámbitos. La
> jurisdicción se protege así tanto en el origen como en el punto de
> consumo. Esta doble barrera reduce la probabilidad de que una
> descripción visual termine compitiendo con una medición verificable.

### Identificar no es medir

> La regla operativa que permite combinar las fuentes se formula como
> «identificar no es medir». La lectura visual puede aportar el
> sustantivo que concreta una afirmación, mientras que el informe aporta
> el valor que la sostiene. Por ejemplo, la lectura puede reconocer a
> «un hombre de espaldas frente al mar» y la métrica puede determinar su
> distancia a un punto de la regla de tercios. El crítico puede integrar
> ambas informaciones en una misma frase siempre que la identificación
> proceda del canal visual y la posición se respalde mediante la cita
> correspondiente.
>
> Esta integración no concede al canal visual autoridad sobre la medida.
> Puede nombrar al hombre, pero no afirmar que está «muy descentrado»;
> puede reconocer un pasillo, pero no declarar que sus líneas convergen
> con fuerza; puede identificar una pared naranja, pero no concluir que
> la imagen emplea correctamente una armonía cromática. Los
> calificativos que gradúan propiedades compositivas siguen necesitando
> evidencia métrica.
>
> La diferencia resulta especialmente visible en imagen30. La condición de aplicabilidad de
> aislamiento se cierra porque la ontología de detección utilizada no
> reconoce el árbol como un sujeto válido. La lectura visual, sin
> embargo, identifica «un gran árbol […] solitario sobre una
> colina». La tercera fuente permite que la crítica explique de qué
> trata la fotografía, pero no reabre las métricas de posición o
> aislamiento asociadas al sujeto. Rellena un hueco semántico sin
> fabricar la evidencia métrica que falta.

### Persistencia, auditabilidad y límites

> La lectura visual es la única fuente del sistema que no puede
> recomputarse de manera determinista. Una temperatura baja reduce la
> variabilidad del modelo, pero no garantiza que dos llamadas produzcan
> exactamente las mismas observaciones. La solución adoptada no consiste
> en presentar esa salida como determinista, sino en persistirla.
>
> La elección de una herramienta independiente, en vez de introducir la
> imagen directamente en el contexto multimodal del crítico, responde
> también a este criterio. Un agente ejecutado mediante un bucle de
> razonamiento podría reenviar la imagen en varias iteraciones. La
> herramienta la procesa una sola vez y devuelve texto al crítico. De
> este modo se acota el coste y, sobre todo, la lectura queda registrada
> fuera del razonamiento interno del LLM.
>
> Las observaciones se almacenan en
> `CriticaCompositiva.observaciones_visuales` mediante objetos con los
> campos `texto` y `ambito`. No incorporan `campos_citados`, porque no
> proceden de una medición. Esta separación permite distinguir
> estructuralmente, y sin tener que interpretar la prosa, la evidencia
> verificable de la información visual no recomputable. El campo
> `vio_imagen` deja registrado si el
> crítico utilizó realmente la herramienta: cuando es falso,
> `observaciones_visuales` debe permanecer vacío.
>
> La auditabilidad de este canal tiene, no obstante, un límite
> explícito. La estructura de la salida permite comprobar que las
> observaciones no incluyan citas, que no aparezcan cuando vio_imagen es
> falso, que no contengan cifras y que su ámbito no coincida mediante
> términos evidentes con una dimensión prohibida. No puede garantizar
> por sí solo que la descripción visual sea verdadera ni detectar todos
> los solapamientos semánticos. Esas dos propiedades requieren revisión
> humana o un corpus anotado. Por ello, la salida se considera
> persistible y revisable, pero no recomputable con las mismas garantías
> que los informes numéricos.

### Límite del canal visual: no puede modificar las métricas

> En el contrato inicial se contempló la posibilidad de que el canal
> visual atenuara una afirmación cuando lo observado no respaldara las
> métricas. Ese mecanismo no forma parte del MVP implementado. La salida
> actual no contiene un bloque de atenuaciones y el crítico tiene
> prohibido rebajar, elevar, corregir o matizar una medición a partir de
> la imagen.
>
> Esta decisión preserva la premisa central de verificabilidad. El
> modelo visual no dispone de una precisión cuantificada comparable con
> la confianza de las métricas, por lo que no existe una regla
> principiada para combinar ambas fuentes sobre una misma variable. Si
> una lectura visual contradice de manera aparente un resultado
> numérico, esa discrepancia debe tratarse como señal para revisar y
> recalibrar el sistema fuera de la ejecución, no como autorización para
> que el LLM modifique la crítica en tiempo real.
>
> La tercera fuente aporta, en consecuencia, amplitud semántica, pero no
> autoridad métrica. Permite que la crítica deje de hablar de figuras
> abstractas y describa una fotografía concreta, manteniendo al mismo
> tiempo una frontera visible entre lo medido y lo observado. Esa
> frontera es la que hace posible incorporar visión generativa sin
> renunciar a la trazabilidad que estructura el resto del sistema.

## Alcance del juicio: qué puede y qué no puede concluir el crítico

> **El problema que resuelve este apartado.**
>
> Los apartados 7.2 y 7.3 fijan cómo escribe el crítico ---qué cita,
> cómo cierra, cómo resuelve una tensión---. Este apartado fija algo
> distinto y más delicado: hasta dónde puede llegar su conclusión antes
> de dejar de ser una crítica anclada en evidencia y convertirse en una
> opinión. En un sistema cuya tesis central es la verificabilidad, la
> frontera del juicio no puede quedar implícita en el estilo de
> redacción: tiene que estar tan delimitada como una condición de aplicabilidad del nivel 2.
>
> **La decisión que se sopesó explícitamente.**
>
> Al diseñar el nivel 3 se valoraron dos alternativas. La primera limitaba al
> crítico a describir lo medido y evitaba la pregunta «¿con qué autoridad concluye
> este sistema que una foto funciona?», pero no cumplía el objetivo de emitir un
> veredicto. La solución intermedia permite dictaminar bajo dos condiciones sin
> excepción.
>
> **Lo único que el crítico puede concluir.**
>
> El sistema permite un solo tipo de veredicto: si la fotografía se
> acoge o no a las convenciones compositivas que más pesan en su
> contexto. Esa conclusión solo es válida si cumple dos condiciones
> simultáneas:
>
> Se apoya en una afirmación ya citada en algún punto anterior de la
> crítica ---no introduce evidencia nueva para cerrar---.
>
> Nombra el contexto que sostiene el peso invocado, de modo que el
> veredicto quede explícitamente condicionado a esa clase de escena y no
> se lea como una regla universal.
>
> El ejemplo que fija el criterio en la propia plantilla de instrucciones es: «en una escena
> de arquitectura la estructura lineal es lo que más cuenta `[W = 0.37]`, y
> aquí las líneas no llegan a organizar la mirada
> `[score_convergencia.valor = 0.31]`». Es una conclusión trazable
> porque cada uno de sus dos componentes ---el peso y la métrica--- ya
> apareció citado en el cuerpo de la crítica; el veredicto no hace más
> que enlazarlos.
>
> **Por qué solo el crítico puede hacer esto.**
>
> La capacidad de hablar de importancia y jerarquía no es una licencia
> estilística concedida al nivel 3: es la consecuencia directa de que es
> el único componente del grafo que consume W. Los cuatro especialistas
> tienen prohibido, por diseño, decir cuánto pesa lo que miden ---cada
> uno dice QUÉ pasa en su dimensión, nunca CUÁNTO importa---,
> precisamente porque el vector de pesos está aislado y nunca llega a su
> contexto de tarea (aislamiento descrito en el apartado 4.6). El crítico es el
> punto exacto del sistema donde ese aislamiento se levanta, y por eso
> es también el único punto donde un juicio de importancia deja de ser
> una alucinación para pasar a ser una lectura del prior contextual que
> el sistema ya declaró.
>
> **Lo que sigue prohibido, sin excepción.**
>
> Tres cosas quedan explícitamente vetadas, y ninguna de las tres se
> relaja aunque el crítico disponga de la lectura visual:
>
> El gusto personal («una foto preciosa», «poco interesante»): no es un
> juicio anclable a ninguna métrica ni a W, es una preferencia estética
> sin evidencia detrás.
>
> El juicio sobre el autor («le faltó oficio», «no supo ver»): el
> sistema evalúa una fotografía, no a quien la hizo.
>
> Concluir sobre una dimensión cuyas métricas están todas cerradas: ahí
> no hay nada que dictaminar, y hacerlo sería inventar exactamente lo
> que el resto del sistema se ha dedicado a impedir.
>
> **Un límite más estructural: nunca hay una nota única.**
>
> El sistema tampoco permite, en ningún nivel, agregar las cuatro
> dimensiones en una puntuación de calidad global. La misma frontera que
> impide ordenar orden_dimensiones por W × cobertura (apartado 7.1) gobierna
> también el veredicto final: no hay un «8/10» ni un índice combinado.
> La razón es la misma en los dos casos ---una cifra única esconde
> justamente la información que toda la arquitectura existe para
> exponer: qué dimensión pesa más en este contexto, cuál quedó sin
> medir, y dónde dos dimensiones tiran en sentidos distintos. Reducir
> eso a un número sería deshacer con una frase final todo el trabajo de
> trazabilidad del resto de la crítica.
>
> **El límite del canal visual sobre el juicio.**
>
> Aunque el crítico disponga de la lectura visual, esa fuente tiene
> prohibido alimentar el veredicto salvo para nombrar lo que una métrica
> ya mide ---la regla de jurisdicción del apartado 7.4.4, «identificar no
> es medir»---. El mecanismo adicional que el contrato original
> contemplaba, un veto asimétrico que habría permitido a la lectura
> visual rebajar una afirmación métrica, quedó fuera del alcance actual
> por los motivos que expone el apartado 7.4.6. Sobre el juicio, la
> consecuencia es directa: si la lectura visual contradice una medición,
> esa contradicción no llega a la crítica; es la señal de recalibrar el
> sistema fuera de línea, exactamente la misma disciplina con la que las
> cuatro revisiones visuales del nivel 2 corrigieron sus condiciones de aplicabilidad en su
> momento, no una vía para que el modelo de lenguaje corrija a la
> métrica en tiempo real.
>
> **Relación con la validación.**
>
> Este veredicto ---anclado, condicionado y sin agregación--- es el que
> se ejercita en la comparación de las dos configuraciones del crítico
> del apartado 9.2.3, donde se comprueba que ver la fotografía no le da
> al modelo una excusa para concluir sobre una dimensión que el nivel 2
> dejó sin medir. La comparación con un modelo generalista con la imagen
> y una instrucción independiente, al margen del sistema ---si una
> conclusión obligada a citar su propia evidencia resulta tan
> convincente para un lector como una conclusión sin esa obligación---,
> queda fuera del alcance de este trabajo y se declara como línea de
> trabajo futuro en el apartado 10.4.

## Interfaz de usuario: entrega de la crítica y de su evidencia

### Alcance declarado

> La interfaz (app.py, Streamlit) responde al cuarto objetivo específico:
> entregar la crítica y su evidencia visual a un fotógrafo, no solo a quien sepa
> ejecutar el sistema desde la terminal. Su alcance funcional comprende una
> imagen a la vez, sin histórico ni comparador.
>
> Se sube una imagen, se analiza, se muestra el resultado y un botón
> reinicia la vista para poder subir otra. Tampoco hay tablas de
> métricas, por el motivo que desarrolla el apartado 7.6.4.

### El principio de reutilización: cero lógica de dominio nueva

> La interfaz no reimplementa nada que el resto del sistema ya resuelva.
> Lanzar el crew es una llamada a `main.construir_inputs(ruta)`, la misma
> función que usa la entrada de terminal (`crewai run`), lo que garantiza
> que cada análisis escriba en su propio directorio sin importar por qué
> vía se lanzó. Saber dónde leer los resultados es una llamada a
> `rutas.dir_ejecucion(ruta)`. Cada uno de los seis JSON ya trae su propio
> `verificacion_path`, así que no hay reconstrucción de rutas por
> convención de nombres. `salida_critica.json` ya trae la sintesis
> redactada, sin nada que formatear. Y como los dos modelos pesados
> (ResNet18, YOLO) se cargan a nivel de módulo dentro de sus
> herramientas, su coste se paga una sola vez al arrancar la aplicación,
> lo que hace innecesario `@st.cache_resource` sobre el crew ---una
> simplificación deliberada frente a un problema que, en este diseño, ya
> no existe---.

### Un indicador de progreso

> La función kickoff() de crewai no informa mientras corre y puede tardar varios
> minutos. La solución es Crew(task_callback=...), inyectado en el método crew()
> de `@CrewBase` para hacer visible el acoplamiento desde crew.py. Tras cada tarea,
> el callback actualiza un `st.status` con una etiqueta legible. Como avisa de lo
> que terminó y no de lo que empieza, el paso en curso se obtiene de la siguiente
> tarea de la lista. Además, se ejecuta antes de que CrewAI vuelque el
> `output_file`, por lo que todavía no puede leer el JSON recién producido.

### Qué se muestra, y qué no

> Se muestran solo tres cosas: la síntesis (el único campo del sistema
> pensado para el usuario final), las salvedades (en un desplegable con
> el recuento en el título, fáciles de mostrar y que explicitan qué no
> puede afirmar) y las imágenes de verificación. La tentación descartada
> de forma expresa es volcar el bloque arbitraje y los 81 campos de los
> cinco informes en tablas: triplicaría el código sin aportar nada,
> porque el argumento de «esto no es una caja negra» ya lo ganan las
> imágenes de verificación ---que se entienden de un vistazo--- y no lo
> gana un JSON.

### Los seis paneles de verificación

> De los ocho artefactos visuales que el sistema genera, la interfaz
> muestra seis: sujeto detectado, mapa de saliencia, composición
> espacial, líneas y dirección, espacio y aislamiento del sujeto, y luz y tono. Se
> dejan fuera dos. El primero es deteccion_yolo_path, porque
> percepcion_bbox_path ya dibuja
> el cuadro delimitador final resuelto y coloreado según su procedencia, y
> enseñar los dos solo repetiría el mismo rectángulo obligando al
> usuario a averiguar en qué se diferencian. El segundo es el mapa
> Grad-CAM del clasificador de contexto: se genera y se persiste en
> outputs/gradcam/ (apartado 5.1.7), pero no llega a la interfaz, porque
> responde a una pregunta distinta de la de los otros seis. Los paneles
> que sí se muestran verifican una medición sobre la fotografía ---dónde
> cae el sujeto, qué líneas concurren, qué separa la figura del fondo,
> qué colores dominan---, y cada uno respalda afirmaciones que el usuario
> acaba de leer citadas en la síntesis. El Grad-CAM, en cambio, explica
> una decisión del modelo: por qué la red clasificó la escena en una
> categoría y no en otra. Es explicabilidad dirigida a quien desarrolla o
> audita el sistema, no a quien quiere entender su fotografía, y por eso
> se conserva como artefacto en disco en lugar de ocupar un panel. El
> requisito que el apartado 5.1 declara innegociable se cumple
> igualmente: pide que toda predicción de clase vaya acompañada de un
> mapa que permita verificar en qué regiones se apoyó la decisión, y eso
> se satisface generándolo y conservándolo, no exhibiéndolo. El usuario
> tampoco queda sin recurso frente a una etiqueta de contexto
> equivocada, porque la síntesis nombra el contexto que sostiene la
> jerarquía del discurso: puede detectarla leyendo, aunque no pueda
> diagnosticarla.
>
> Cada panel lleva, además de
> su título, una frase fija de contexto ---por ejemplo, para el mapa de
> saliencia: «Dónde se concentra el peso visual de la escena: cuanto más
> claro, más atrae la mirada esa zona. De este mapa sale el centro de
> masas con el que se mide el equilibrio de la composición»---, porque
> un rótulo por sí solo no dice nada a quien no ha leído el código, y
> ese panel en concreto ---una imagen en escala de grises sin nada
> dibujado encima--- resulta difícil de interpretar sin ella.

### La leyenda estática de escalas

> Al final de la vista de resultados hay una leyenda plegada que explica
> en qué escala vive cada métrica citable, para que una cifra como
> `[ratio_nitidez.valor = -0.9124]` en la síntesis signifique algo para
> quien no ha visto el código. Su contenido está transcrito directamente
> de los bloques ESCALA DE LECTURA de schemas/especialistas.py, que son
> escalas absolutas derivadas del dominio de cada magnitud ---no rangos
> observados sobre data/--- y por tanto no caducan al crecer el corpus.
> Es texto fijo que no lee ningún JSON, así que por construcción no
> puede contradecir a ningún informe. La única excepción marcada es
> d_equilibrio, etiquetada como «referencia orientativa» y no como
> umbral: el sistema emite deliberadamente valor_norm = None en ese
> campo porque su cota teórica no es interpretable, y presentar en la
> interfaz un umbral que el propio sistema se niega a afirmar
> internamente rompería la coherencia de todo el diseño.

### Robustez frente a fallos parciales

> leer_json devuelve un diccionario vacío si el fichero falta o está
> corrupto y la vista simplemente se salta ese panel. Si el crew cae a
> mitad de la ejecución, los informes de las tareas que sí terminaron
> quedan escritos y se muestran igualmente.

### Dónde se guarda la imagen de subida

> st.file_uploader entrega bytes en memoria y un nombre, nunca una ruta
> ---el navegador no la envía, por seguridad---. Por ello se vuelca en
> outputs/subidas/ para que el flujo de procesamiento, basado en rutas, pueda
> analizarla. Se conserva su nombre original, no uno aleatorio: el stem del fichero
> es la clave primaria de todo el sistema ---el directorio de ejecución,
> los cinco PNG de verificación, Grad-CAM, la detección YOLO, el mapa de
> saliencia, la caché de outputs/lectura_visual/ y la guarda anti-cruce
> de PrioridadTool descrita en el apartado 7.1---. Con un nombre aleatorio esas
> carpetas se llenarían de ficheros ilegibles y la caché de la lectura
> visual no acertaría nunca, lo que costaría una llamada a Gemini en
> cada reanálisis de la misma fotografía. El precio aceptado a cambio, y
> es una decisión consciente para un sistema de un solo usuario en
> local: dos imágenes subidas con el mismo nombre se pisan.

# Validación experimental: diseño e instrumentos

Los capítulos 5, 6 y 7 describen qué mide y cómo razona el sistema. Este capítulo
establece cómo se comprueba su fiabilidad y se separa deliberadamente del 9: aquí
se fija el diseño de cada
experimento y el instrumento que lo mide, antes de presentar ningún
resultado. En un sistema con plantillas de instrucciones y métricas calibradas sobre un
corpus propio, describir el protocolo de medida después de conocer el
resultado invita a elegir el umbral que mejor cuadre.

La validación de este trabajo opera en varios niveles, y cada uno
necesita un instrumento distinto porque mide una propiedad distinta del
sistema:

- Se comprueba la corrección aritmética de cada motor de cómputo.

- Se comprueba la validez de la calibración de cada condición de aplicabilidad.

- Se comprueba la fidelidad de la crítica textual respecto a la evidencia numérica que
  la sostiene.

- Se mide el valor añadido de la tercera fuente, la lectura visual, frente al mismo
  crítico sin ella.

## Qué se puede validar de un juicio estético y qué no

> Antes de diseñar hay que fijar un umbral de frontera, ¿cómo se sabe
> que la crítica que produce el sistema es correcta? La respuesta
> honesta empieza por reconocer que esa pregunta no tiene un
> instrumento de medida posible, y que este trabajo no pretende
> demostrar. Por qué el juicio estético en sí no es validable. No existe
> un valor objetivo para «esta fotografía está bien». Los conjuntos de
> datos que se usan habitualmente para entrenar modelos de estética
> automática como AVA, EVA o NIMA no contienen una verdad objetiva, sino
> un consenso estadístico de puntuaciones humanas, que varía por
> cultura, época y panel de evaluadores. Un sistema que aprendiera a
> reproducir esa media estaría, como mucho, validado contra el consenso
> de un panel concreto, no contra la corrección de juicio. Este proyecto
> no dictamina calidad estética incondicional. El único juicio de valor
> que el crítico tiene permitido emitir es uno anclado, nunca un
> veredicto de gusto ni un juicio sobre el autor. En consecuencia, lo
> que hay que validar no es si el veredicto es «correcto», porque esa
> pregunta no tiene respuesta bien definida, sino si el sistema respeta
> su propio contrato: que no afirma más de lo que sus métricas pueden
> sostener.
>
> Lo que sí es validable, y es lo que estructura el resto del capítulo.
> Reformulada así, la pregunta se descompone en cuatro propiedades
> independientes, verificables cada una con su propio instrumento:
>
> ¿Calcula bien lo que dice calcular? Es la corrección de cada motor
> (NumPy/OpenCV/scikit-learn) frente a un resultado analítico conocido
> de antemano. No es una pregunta estética, es aritmética verificable, y
> por eso se responde mediante nueve pruebas analíticas: el proyecto
> fabrica escenas sintéticas con la geometría
> exacta que el motor debe recuperar (p. ej., un punto de fuga trazado
> a mano en (0,6836; 0,3660) y recuperado en (0,6812; 0,3662)) y
> comprueba que el motor recupera ese valor de referencia: de forma
> exacta cuando la magnitud lo permite ---un área de máscara, un matiz
> fabricado a un ángulo concreto--- y dentro de una tolerancia declarada
> cuando intervienen redondeos de rejilla, como en el punto de fuga del
> ejemplo, recuperado a unos 2,5 px sobre un lienzo de 1024.
>
> ¿Declara bien cuándo no puede afirmar nada? Es la validez de las
> condiciones de aplicabilidad: si `confianza = 1`, ¿hay de verdad algo que medir?; si
> `confianza = 0`, ¿es cierto que la escena no lo permite? Se mide con revisión
> visual imagen a imagen (el método que ya recalibró tres de las cuatro
> condiciones de aplicabilidad del sistema tras encontrar falsos positivos que ninguna prueba automatizada
> analítica podía detectar) y con la existencia de una banda vacía en el
> corpus que separe los dos casos.
>
> ¿La crítica dice solo lo que puede probar? Es la fidelidad de la parte
> generativa (el LLM) frente a la parte determinista (el informe): que
> cada cifra citada exista en el informe con ese valor, que ninguna
> afirmación cite una métrica cerrada, y qué fracción de la síntesis
> descansa en algo verificable frente a algo que no lo es (la cobertura
> de trazabilidad). Se comprueba a mano y sin ningún modelo de lenguaje
> adicional. El instrumento ya detectó un defecto real de plantilla de instrucciones
> sobre `imagen18`; el apartado 9.4 presenta el resultado de corregirlo. Las dos
> ejecuciones se conservan archivadas ---el antes en
> `outputs/ablacion_salvedades/antes/imagen18` y el después en
> `outputs/validacion_2026-08-21/imagen18`---, de modo que la comparación
> puede rehacerse sin volver a ejecutar el sistema. Esa garantía cubre
> este experimento, pero no el de las dos ramas del crítico del apartado
> 9.2.3, cuyas ejecuciones se sobrescriben entre sí y del que solo se
> conserva una rama por imagen.
>
> ¿Aporta algo la tercera fuente, y a qué precio? Es la única pregunta
> de las cuatro que roza el terreno estético, y se acota con cuidado
> para no caer en el problema del punto anterior: no se mide «qué
> crítica es más verdadera» (no hay verdad de referencia), sino qué
> añade a la síntesis que el crítico vea la fotografía y si eso cuesta
> algo en disciplina de cierre o en trazabilidad ---dos propiedades que
> no dependen de preferencias subjetivas---. Comparar la tercera fuente con
> un modelo generalista con la imagen y una instrucción independiente, al margen del
> sistema, que sería la otra forma de responder a «¿aporta algo frente
> a la alternativa generalista?», exigiría un panel y una rúbrica ciega:
> queda fuera del alcance de este trabajo y se declara como línea de
> trabajo futuro en el apartado 10.4.

## Conjuntos de imágenes: conjunto de prueba de EVA, corpus de desarrollo

> Este trabajo no usa un único banco de imágenes para validarse: usa
> tres, y cada uno cumple un papel que los otros dos no pueden cumplir.
> Confundirlos sería un error metodológico serio.
>
> El conjunto de prueba de EVA tiene una estructura de partición fija, disjunta del
> entrenamiento y usada una vez para reportar la cifra final del
> clasificador de contexto. Procede del conjunto de datos EVA (Kang, Valenzise,
> Dufaux, 2020), 5101 imágenes en total, cuyas seis categorías se
> asignaron mediante el procedimiento híbrido ---categorización
> automática con YOLOv3 y revisión humana posterior--- que describe el
> apartado 5.1.1.
>
> La partición es estratificada por esas cinco clases, con un cuarto
> conjunto reservado aparte para calibrar el rechazo, como recoge la
> Tabla 8.1.

**Tabla 8.1.** Partición del conjunto de datos EVA por categoría:
entrenamiento, validación, prueba y la reserva `otro` para calibrar el
rechazo.

  --------------------------------------------------------------------------
  Partición   animal   arquitectura   paisaje   producto   retrato   Total
  ----------- -------- -------------- --------- ---------- --------- -------
  Entrenamiento 634      663            708       649        724       3378

  Validación   136      142            152       140        155       725

  Prueba      136      143            152       139        155       725

  Otro        \-       \-             \-        \-         \-        273
  (rechazo)
  --------------------------------------------------------------------------

> La partición de prueba (725 imágenes, la tabla anterior) es la que sostiene la matriz
> de confusión y el F1 por clase del apartado 9.1; `otro` (273 imágenes,
> categoría EVA reservada y nunca vista en entrenamiento) es la que
> sostiene la calibración del umbral de rechazo, cuyo valor y cuya
> justificación se presentan en ese mismo apartado. El resultado queda
> además versionado dentro de este repositorio, en
> `models/clasificador_contexto/umbral.json`, junto con el análisis
> cuantitativo que lo respalda.
>
> El corpus de desarrollo, ubicado en `data/`, contiene 40 imágenes a fecha de esta
> redacción y sigue creciendo. A diferencia del conjunto de prueba de EVA, no es un
> banco de validación estadística: no hay partición de entrenamiento/prueba, ninguna
> imagen se reserva «para el final», y el propio sistema no lo lee
> nunca en producción
> --- lo recorren solo los programas de calibración de `tests/`. Su función
> es otra: es el corpus sobre el que se calibran los umbrales de las
> condiciones de aplicabilidad del nivel 2 (p. ej. `LONGITUD_MIN_HORIZONTE = 0.13`,
> `RATIO_CROMATICO_MIN = 0.10`) mediante el ciclo descrito en el apartado 3.3.1 ---
> contrato provisional → sonda de calibración → revisión visual del autor → contrato
> cerrado ---, y sobre el que se miden los recuentos que se citan en el
> capítulo 9 («la condición de aplicabilidad de convergencia abre en 7 de 40»).
>
> En toda la memoria, el corpus de desarrollo son las 40 imágenes de
> `data/`, y cada cifra experimental declara sobre cuántas se midió.
> Los recuentos de barrido ---los que se obtienen recorriendo el corpus
> entero con uno de los motores--- están todos tomados sobre esas 40.
> Junto a ellos aparecen dos denominadores distintos, y los dos por un
> motivo explícito. El primero es el corpus de aplicabilidad: el
> protocolo del apartado 8.4.2 excluye `imagen7` por no ser una fotografía
> y se aplica sobre 39, de modo que una cifra «sobre 39» de ese
> protocolo no es comparable con un recuento de barrido. El segundo son
> unas pocas mediciones puntuales que no son barridos ---la
> comprobación de determinismo de un algoritmo, la calibración original
> de un umbral--- y que no podrían repetirse hoy sin desactivar una
> semilla o un redondeo del propio sistema; esas conservan el número de
> imágenes sobre el que se tomaron, 25 o 29, y lo declaran en la misma
> frase.
>
> Dos matices hay que declarar con honestidad porque afectan a la
> validez externa
> (apartado 9.5): dos de las cuarenta imágenes (`imagen7`, `imagen10`) no son
> fotografías reales sino una captura de pantalla y una pintura
> abstracta, ambas con marca de agua, y se conservan a propósito porque
> son justo el tipo de entrada donde las condiciones de aplicabilidad deben cerrarse ---de
> hecho el clasificador de contexto rechaza `imagen7` y la resuelve como
> `otro`, que es el comportamiento que se le pide ante una entrada ajena
> a las cinco clases---; y existe
> una regla explícita de integridad del corpus --- añadir imágenes es seguro,
> sustituir o borrar las existentes invalida en silencio los barridos
> anteriores, porque las cifras ya escritas pasarían a describir otra
> fotografía.

## Instrumentos de medida

> Ningún instrumento de validación usa un modelo de lenguaje para juzgar a otro.
> Hay dos tipos, y no son intercambiables. Uno comprueba que
> las fórmulas calculan lo que dicen calcular, sobre casos donde la
> respuesta correcta se conoce de antemano. El otro comprueba que la
> crítica final no dice más de lo que sus datos permiten. El primero es
> aritmética; el segundo es honestidad.

### Revisión manual dirigida, sin modelos de lenguaje

> La pregunta que responde este instrumento es la siguiente:
> ¿cómo se sabe que la crítica no se inventa cosas? La respuesta no
> puede ser «porque confiamos en el LLM» --- eso sería exactamente la
> caja negra que el trabajo intenta evitar. Y tampoco puede ser «porque
> otro LLM lo revisó», porque entonces solo habría trasladado el
> problema de fiabilidad de un modelo generativo a otro, sin ganar nada
> verificable.
>
> El instrumento consiste en una lista de comprobación
> fija de cuatro preguntas que se aplican a mano sobre una ejecución ya
> archivada, cotejando el texto que escribió el crítico con los informes
> numéricos que lo sostienen. No juzga si la crítica es acertada ---eso,
> como se argumenta en el apartado 8.1, no tiene una respuesta
> objetiva---, sino si es honesta:
>
> 1\. ¿Se cita el valor de alguna métrica que el propio sistema declaró
> no aplicable?
>
> 2\. ¿Toda cifra del texto se recompone desde un campo del informe,
> admitiendo solo las transformaciones exactas que el sistema autoriza
> (el redondeo a cuatro decimales y el complementario 1 − v)?
>
> 3\. ¿Alguna observación visual invade una de las cuatro dimensiones
> medidas?
>
> 4\. ¿El discurso sigue el orden de dimensiones que fijó el arbitraje,
> y no el que le habría convenido al modelo al redactar?
>
> Las cuatro se responden consultando dos ficheros y ninguna necesita un
> modelo de lenguaje, que es lo que las hace repetibles por un tercero.
> Su contrapartida es el tamaño de la muestra: revisar a mano, campo a
> campo, obliga a primar la profundidad del caso sobre el número de
> casos, y eso condiciona el protocolo del apartado 8.4.3.
>
> Las dos primeras preguntas permiten obtener una magnitud que el sistema no
> registraba: qué fracción de la crítica final descansa en algo verificable
> frente a lo que no lo es ---la cobertura de trazabilidad---. Se cuenta
> sobre la síntesis y frase a frase: el numerador son las frases que
> contienen al menos una cita verificada contra su informe, y el
> denominador, todas las frases de la síntesis. Las frases sin cifra
> ---una observación visual, una salvedad--- suman en el denominador y no
> en el numerador, de modo que la cobertura baja cuando la crítica
> describe más de lo que mide, que es exactamente el efecto que interesa
> vigilar. Esta medida permite evaluar si la ampliación del canal visual
> cuenta con respaldo verificable.

### Pruebas analíticas con valor de referencia conocido y sondas de calibración

> Aquí el problema de partida es distinto: no hay ningún banco de datos
> que diga «esta fotografía tiene el punto de fuga exactamente aquí» o
> «esta composición está a tantos puntos de los tercios». Para poder
> comprobar que una fórmula calcula bien algo así, se construye el caso
> de forma controlada: se dibuja una escena cuya respuesta correcta se
> conoce de antemano porque se ha definido geométricamente, no porque se
> haya medido, y se comprueba que el sistema
> llega al mismo sitio. Es el mismo principio en los cuatro
> especialistas: se fabrican mapas, máscaras o colores con una propiedad
> exacta conocida (un color puesto a un ángulo concreto, un punto de
> fuga trazado antes de que el algoritmo lo busque, una zona de interés
> colocada en un punto exacto de la imagen), y se comprueba que el motor
> de cálculo la recupera. Cuando el algoritmo de base (segmentación,
> agrupamiento de color) no se puede controlar directamente, se aplica el
> mismo principio de forma inversa: se construye la entrada de forma que solo
> pueda dar un resultado razonable, y se comprueba eso.
>
> Esta batería cubre cada componente que produce un número (el
> clasificador, la percepción compartida, los cuatro especialistas, el
> vector de pesos y la parte determinista del crítico), y todos
> comparten la misma lógica: no comprueban si el resultado «parece
> correcto», comprueban si coincide con un valor que se fijó antes de
> ejecutar nada.
>
> Las sondas de calibración son una herramienta distinta y con otro propósito: no
> comprueban, miden. Son el instrumento del ciclo ya descrito en el apartado 3.3.1
> ---contrato provisional, sonda de calibración, revisión visual del autor, contrato
> cerrado---: recorren el corpus real y producen la evidencia (tablas,
> gráficas, casos límite) sobre la que luego se decide, mediante revisión
> visual, dónde
> poner un umbral. No determinan un aprobado o un fallo; proporcionan
> información para la decisión.
>
> La distinción entre las dos cosas explica una situación relevante:
> las pruebas automatizadas pueden seguir superándose aunque la calibración que sostienen se haya quedado
> desfasada, porque comprueban que una fórmula sigue calculando bien
> ---eso no cambia nunca---, no que un umbral siga separando bien dos
> tipos de fotografía en un corpus que ha crecido. Solo una sonda de calibración,
> revisada de nuevo, permite comprobarlo. Por este motivo, un umbral
> calibrado sobre un corpus pequeño necesita revisarse cada vez que el
> corpus crece, aunque todas las pruebas automatizadas sigan pasando.

## Protocolos experimentales

> Esta sección fija qué se va a medir, sobre qué datos y con qué
> criterio de éxito, para cada uno de los cuatro experimentos.

### Clasificador de contexto

> La evaluación del clasificador de contexto no requiere ninguna
> ejecución adicional: se traslada la obtenida en el proyecto externo de
> entrenamiento
> (`Curso_CNN_FineTunning/clasificador_contexto/notebooks/fase5_evaluacion.ipynb`)
> sobre el conjunto de prueba de la partición estratificada de EVA
> (3378/725/725), calculando el informe de clasificación estándar
> (precisión, exhaustividad y F1 por clase) y la matriz de confusión
> 5×5. El análisis del error dominante se completa localizando en la
> matriz el par de clases con mayor confusión y examinando su
> explicabilidad mediante Grad-CAM sobre ejemplos concretos mal
> clasificados de ese par.

### Revisión de las condiciones de aplicabilidad

> Para cada una de las 40 imágenes del corpus y cada especialista, se revisa el
> panel completo, con todas sus métricas y condiciones a la vez. Se asigna un
> veredicto de tres niveles: ✓ el
> panel es compositivamente correcto --- incluyendo los casos en los que
> una condición de aplicabilidad cierra y hacerlo es lo acertado, no solo aquellas que abren bien
> ---; ≈ una parte del panel es correcta y otra no (p. ej. una de las
> dos métricas del especialista acierta y la otra no); ✗ el panel no es
> compositivamente defendible. `imagen7` queda sin evaluar porque es una
> captura de pantalla del explorador de Windows, no una fotografía
> (deuda ya documentada), reduciendo el corpus efectivo de este
> protocolo a 39 imágenes. Ese subconjunto ---las 40 del corpus de
> desarrollo menos `imagen7`--- es el que en adelante se llama corpus de
> aplicabilidad, y es el único denominador de 39 que aparece en la
> memoria: todos los recuentos de barrido se dan sobre las 40.
> `imagen10` sí se evalúa pese a ser una pintura
> abstracta y no una fotografía real, porque el sistema la procesa igual
> que cualquier imagen.

### Fidelidad métrica-texto y cobertura de trazabilidad

> Las cuatro preguntas del apartado 8.3.1 se aplican a mano sobre dos
> ejecuciones archivadas, cada una en las dos ramas del crítico. La muestra se
> limita a cuatro ejecuciones porque la revisión se hace campo a campo y prima la
> profundidad sobre el número de casos. Se eligen `imagen30`, donde 7 de las 10 métricas del nivel 2 están
> cerradas y es especialmente relevante comprobar que las observaciones
> visuales no sustituyen a las métricas, e `imagen4`, con un perfil de cierre
> intermedio que sirve de contraste. Contrastar cada una con el indicador booleano de lectura visual
> en ambos estados es lo que permite comprobar si activar la tercera
> fuente introduce alguna violación que no existiera sin ella.

### Comparación con y sin lectura visual

> El crítico del sistema (nivel 3) puede operar en dos configuraciones,
> controladas por una sola constante (`FLAG_CRITICO_VE_IMAGEN`). En la
> rama sin lectura visual arbitra la crítica usando los cuatro
> informes de los especialistas y el vector W, sin ver la
> fotografía. En la rama con lectura visual conserva esas fuentes y
> añade una lectura de la imagen hecha por un modelo multimodal, ciega a
> las métricas y limitada a describir contenido que ninguna métrica
> mide.
>
> El sistema se reejecuta sobre cinco imágenes, alternando únicamente
> `FLAG_CRITICO_VE_IMAGEN`. Cubren cinco de las seis
> etiquetas de contexto: `imagen30` (paisaje), `imagen4` (retrato humano),
> `imagen29` (animal), `imagen_puente` (arquitectura) e `imagen_jarron_blanco`
> (producto y bodegón). Las tres primeras proceden del corpus de
> desarrollo; las dos últimas son fotografías sueltas que no forman
> parte de `data/`, y por eso sus nombres no siguen la numeración del
> corpus. Antes de comparar las salidas se verifica que los informes
> numéricos de los cuatro especialistas y el arbitraje completo sean
> idénticos entre las dos configuraciones.
>
> La comparación comprueba cuatro aspectos: si se cita el valor de una
> métrica cerrada, si cambia la cobertura de trazabilidad, si la lectura
> visual invade alguna de las cuatro dimensiones medidas y si la rama
> con lectura visual incorpora contenido identificable que no procede de
> las métricas ni de la etiqueta de clase. Con cinco pares de ejecuciones,
> el análisis es exploratorio y no permite separar por completo el efecto
> del canal visual de la variabilidad ordinaria de la prosa generada.

# Resultados y discusión

Este capítulo sigue el orden de los protocolos del capítulo 8: evaluación del
clasificador (9.1); aplicabilidad, fidelidad métrica-texto y comparación del
crítico con y sin lectura visual (9.2); coste y recursos (9.3); discusión de lo
que demuestran los resultados (9.4); y limitaciones (9.5).

## Clasificador de contexto: precisión, F1 por clase, matriz de confusión y análisis del error dominante con GradCAM

> **Protocolo de evaluación.** El clasificador (`ResNet18` con ajuste fino en
> dos etapas sobre el conjunto de datos EVA) se evalúa sobre `test.csv` --- 725
> imágenes---, la única de las tres particiones (entrenamiento/validación/prueba) que no
> influyó en el entrenamiento ni en la selección de hiperparámetros. `val.csv`
> había guiado
> la elección del mejor punto de guardado entre la Fase 3 (extractor convolucional
> congelado, `val_acc` ≈ 0,84) y la Fase 4 (ajuste fino ligero de `layer4`,
> `val_acc` ≈ 0,85), por lo que una evaluación sobre `val.csv` habría estado
> contaminada por esa misma decisión. Se carga el punto de guardado de la Fase
> 4 y se evalúa en modo `eval()` sin aumento de datos.
>
> **Métricas globales y por clase.** La exactitud sobre el conjunto de prueba es 0,8510 (617
> de 725 imágenes correctamente clasificadas), ligeramente por debajo
> del val_acc de referencia --- coherente con que el conjunto de prueba contiene imágenes
> que el modelo no vio ni en el entrenamiento ni en la selección de
> punto de guardado. El desglose por clase
> (`sklearn.metrics.classification_report`) es el de la Tabla 9.1:

**Tabla 9.1.** Precisión, `recall`, F1-score y soporte por clase sobre
las 725 imágenes del conjunto de prueba de EVA.

  ---------------------------------------------------------------
  Clase                 Precisión   Recall   F1-score   Soporte
  --------------------- ----------- -------- ---------- ---------
  Animal                0,9191      0,9191   0,9191     136

  Arquitectura          0,7922      0,8531   0,8215     143

  Paisaje               0,8049      0,8684   0,8354     152

  Producto/bodegón      0,7984      0,7410   0,7687     139

  Retrato/humano        0,9507      0,8710   0,9091     155

  Exactitud global                           0,8510     725
  ---------------------------------------------------------------

> Las clases animal y retrato/humano son las mejor discriminadas
> (F1 = 0,9191 y 0,9091 respectivamente), mientras que producto/bodegón
> es la más débil (F1 = 0,7687, `recall` = 74,1 %). La matriz de
> confusión de la Figura 9.1 desglosa de dónde salen esas cifras:
>
> ![](docs/memoria/media/media/image5.png){width="4.864583333333333in"
> height="4.791666666666667in"}
>
> **Figura 9.1.** Matriz de confusión del clasificador de contexto
> sobre las 725 imágenes del conjunto de prueba de EVA (verdadero en
> filas, predicción en columnas).
>
> La celda más alta fuera de la diagonal es producto → arquitectura, con
> 18 casos (12,9 % de las 139 imágenes reales de producto). Le siguen
> arquitectura → paisaje, con 15 (10,5 %), y su inversa, con 11. El error
> dominante es sistemático y no ruido disperso: la confusión
> se concentra en un par de clases concreto, lo que motiva el análisis
> dirigido con Grad-CAM. El patrón es además consistente con el
> contenido visual de EVA: escenas urbanas con vegetación, jardines con
> elementos construidos o bodegones con fondos texturizados comparten
> estadísticas de bajo nivel con fondos arquitectónicos, y son
> ambigüedades razonables incluso para un observador humano, de modo que
> el patrón es compatible con un solape entre categorías del propio
> conjunto de datos y no necesariamente con un fallo del modelo.
> Distinguir ambas cosas exigiría reanotar esas imágenes, lo que queda
> fuera del alcance de este trabajo.
>
> **Calibración del umbral de rechazo para «otro».** Como el modelo solo
> conoce las cinco clases anteriores, la categoría «otro» no se entrena:
> se calibra un umbral sobre la probabilidad máxima del softmax, usando
> 273 imágenes reservadas (otro.csv) que nunca participaron en
> entrenamiento ni evaluación. El histograma de probabilidad máxima
> entre el conjunto de prueba y «otro» mostró un solapamiento fuerte, especialmente en la
> zona de alta confianza (>0,90): el modelo exhibe el problema clásico
> de sobreconfianza sobre datos fuera de distribución, clasificando con
> seguridad casi absoluta incluso imágenes que no pertenecen a ninguna
> de sus cinco clases. Dos factores lo explican: (a) otro.csv es un caso
> de near-OOD --- mismo origen y estética que las cinco clases
> entrenadas, solo que sin encajar limpiamente en ninguna categoría de
> contenido, el escenario más difícil para métodos basados en softmax; y
> (b) parte de esas 273 imágenes probablemente sí contienen contenido de
> una de las cinco categorías y solo cayeron en «otro» por el criterio
> de anotación original del conjunto de datos, no por un fallo del modelo.
>
> Se consideraron alternativas más sofisticadas (`MaxLogit`, `energy score`,
> ODIN), pero no se implementaron ni compararon experimentalmente; se
> descartaron por alcance, dado el tiempo disponible y que el escenario
> aquí es near-OOD, el más desfavorable para cualquier método basado en la
> salida del clasificador. Con un barrido de umbrales de
> 0,70 a 0,95 y análisis de ganancia marginal (Δ detección de «otro» / Δ
> rechazo falso del conjunto de prueba por tramo), el óptimo puramente cuantitativo se
> sitúa en 0,80 (razón beneficio/coste 2,94). Se elige sin embargo
> `umbral_otro = 0.85`, por encima de ese óptimo estricto, justificado por
> una asimetría de coste propia de la arquitectura del sistema: en este
> proyecto, rechazar una imagen válida no descarta el análisis, sino que
> degrada el vector de pesos contextuales W al reparto uniforme (0,25 en
> cada dimensión) en lugar de la fila específica del contexto --- el
> coste real de un falso rechazo es menor que en un sistema de detección
> de anomalías típico, lo que justifica estirar el umbral mientras el
> tramo siga compensando (razón > 1; en 0,85 la razón es 1,70, frente a
> <1,35 en 0,90--0,95). Con ese umbral: 27,4 % de las imágenes de prueba
> quedarían mal rechazadas como «otro», y 64,8 % de las imágenes de
> «otro» quedarían correctamente detectadas.
>
> **Análisis del error dominante con Grad-CAM.** Sobre el punto de guardado de la
> Fase 4, con Grad-CAM aplicado al último bloque de layer4 (la capa más
> profunda que conserva estructura espacial), se aprecian ---sobre los
> ejemplos inspeccionados: una imagen representativa por clase y casos mal
> clasificados del par dominante--- dos regímenes de activación distintos
> según la clase:
>
> En las clases centradas en un sujeto recortable (animal, producto,
> retrato), el mapa de calor se concentra sobre el objeto principal ---
> texturas y rasgos faciales en animal, rostro en retrato, silueta del
> objeto en producto --- descontando el fondo.
>
> En las clases de escena (arquitectura, paisaje), el calor se
> distribuye por patrones geométricos globales (líneas de fuga,
> transición cielo-tierra, estructuras del entorno), no sobre un sujeto
> único --- coherente con que estas categorías se definen por la
> composición completa de la escena y no por un objeto aislable.

## Validación de los especialistas y del canal visual

> Este apartado reúne los resultados de los tres protocolos que el capítulo 8
> fija sobre el nivel 2 y el nivel 3, en el mismo orden: la revisión de las
> condiciones de aplicabilidad sobre el corpus de aplicabilidad (protocolo del
> apartado 8.4.2), la fidelidad entre métrica y texto con la cobertura de
> trazabilidad (8.4.3) y la comparación del crítico con y sin lectura visual
> (8.4.4). Los tres se responden a mano, con la lista de comprobación de cuatro preguntas
> del apartado 8.3.1.

### Aplicabilidad de las condiciones y revisión de falsos positivos

> La revisión visual de los paneles produjo los resultados de la Tabla
> 9.2. Son el
> veredicto del propio autor sobre los
> paneles de su sistema, emitido por un único evaluador, sin ciego y sobre
> el mismo corpus con el que se calibraron los umbrales que se están
> juzgando. No son, por tanto, una medida comparable con la exactitud del
> clasificador del apartado 9.1, que se obtiene sobre una partición
> reservada: son el instrumento que el apartado 8.3.1 declara, con el
> alcance que ahí se le reconoce y que la séptima limitación del apartado
> 9.5 recoge.

**Tabla 9.2.** Paneles correctos, parciales e incorrectos por
especialista, revisados visualmente sobre las 39 imágenes del corpus de
aplicabilidad.

  ---------------------------------------------------------------------
  Especialista   Correcta   Parcial   Incorrecta   Paneles correctos
  -------------- ---------- --------- ------------ --------------------
  A1             39         0         0            39/39 (100 %)

  A2             34         4         1            34/39 (87,2 %)

  A3             30         5         4            30/39 (76,9 %)

  A4             37         2         0            37/39 (94,9 %)
  ---------------------------------------------------------------------

> (sobre las 39 imágenes del corpus de aplicabilidad definido en el
> apartado 8.4.2)
>
> **Detalles de los casos no perfectos**, recogidos en la Tabla 9.3.

**Tabla 9.3.** Número de imagen de cada panel parcial o incorrecto de
la Tabla 9.2, por especialista.

  ------------------------------------------------------
  Especialista      Parcial             Incorrecta
  ----------------- ------------------- ----------------
  A2                8, 9, 13, 38        32

  A3                14, 24, 26, 37, 39  8, 12, 34, 35

  A4                30, 39              
  ------------------------------------------------------

> **Análisis por especialista.**

- A1 (composición espacial) no deja ningún panel incorrecto. La revisión respalda
  que corregir su condición de
  aplicabilidad a `fuente == "yolo"` ---tras encontrar 16 falsos positivos de 19 en
  `fuente = "saliencia"`--- resolvió el problema de fondo en lugar de desplazarlo.

- A2 (líneas y dirección): `imagen32` es el único ✗, el falso positivo de
  convergencia
  (producto sin perspectiva real, `apertura_haz` = 89,3°) que motivó la nota de
  deuda ya anotada. La revisión visual señala esa misma fotografía. No es una
  comprobación independiente ---el caso
  ya estaba documentado y el evaluador es el mismo---, pero sí confirma que el
  falso positivo es visible en el panel y no solo en la cifra de apertura del
  haz. Los cuatro ≈ (8, 9, 13, 38) son candidatos a revisar
  con más detalle --- `imagen9` ya estaba señalada como pendiente de revisión
  en el cierre de la Fase 2 del Agente 2, así que coincide con una duda ya
  anotada.

- A3 (espacio y aislamiento del sujeto) es el especialista con menos paneles
  correctos. Es también el que acumula más
  limitaciones estructurales ya aceptadas por escrito (clases ausentes
  de COCO para edificios y árboles, la varianza del Laplaciano midiendo
  densidad de detalle y no enfoque, la inestabilidad de GrabCut ante
  desplazamientos sub-píxel). Los cuatro ✗ (`imagen8`, `imagen12`,
  `imagen34`, `imagen35`) se revisaron visualmente uno a uno, y los cuatro
  corresponden a modos de fallo ya documentados y aceptados por escrito
  ---`imagen12`, por ejemplo, es una fachada, y la condición de aplicabilidad cierra por la
  ausencia de una clase «edificio» en COCO, que es un falso negativo
  aceptado explícitamente---, no a un modo de fallo distinto no visto
  antes.

- A4 (luz y tono) da 37 paneles correctos de 39, con `imagen30` e `imagen39`
  como únicos casos parciales. Esta última hizo perder a
  `RATIO_CROMATICO_MIN = 0.10` su
  margen por arriba (mide 0,1029,
  abre la condición de aplicabilidad por solo 0,0029) --- que la revisión visual la marque
  como «a medias» es coherente con que su clasificación esté al borde
  de un umbral frágil, y refuerza la recomendación ya anotada de
  recentrar ese corte.

### Fidelidad métrica-texto y cobertura de trazabilidad

> La aplicación manual de las cuatro preguntas del apartado 8.3.1
> produjo los resultados de la Tabla 9.4.

**Tabla 9.4.** Resultado de las cuatro preguntas de la revisión manual
dirigida (apartado 8.3.1) sobre `imagen4` e `imagen30`, con y sin
lectura visual.

  --------------------------------------------------------------------------------------------------
  Ejecución   P1 - ¿cita P2 - ¿toda cifra trazable    P3 - ¿la      P4 - ¿el discurso sigue
              métrica    con transformación exacta?    lectura       orden_dimensiones?
              cerrada                                 visual invade 
              por su                                  una dimensión 
              valor?                                  medida?       
  ----------- ---------- ---------------------------- ------------- --------------------------------
  Imagen4     No --- 0/3 Sí --- 8/8 cifras            --- (sin      Sí ---
  (sin        cerradas   verificadas (redondeo        lectura       luz→espacio→composición→líneas
  visión)     citadas    correcto, no truncado)       visual)       

  Imagen4     No --- 0/3 Sí --- mismas 8 cifras, sin  Casi: «fondo Sí, mismo orden
  (con        cerradas   alterar                      de color      
  visión)     citadas                                 claro» roza
                                                      el            
                                                      vocabulario   
                                                      de luz/color  
                                                      sin citar ni  
                                                      contradecir   
                                                      ninguna       
                                                      métrica ---   
                                                      nota menor,   
                                                      no infracción 

  Imagen30    No --- 0/7 Sí ---                       --- (sin      Sí --- luz→composición→(espacio
  (sin        cerradas   d_equilibrio.valor=0.0993,   lectura       y líneas, ambas cerradas)
  visión)     citadas,   media_L.valor=54.5853,       visual)       
              el caso de coord_centro_masa redondeado                 
              más riesgo correctamente a 0,6578 y no                 
              del corpus truncado a 0,6577                                 

  Imagen30    No --- 0/7 Sí --- mismas cifras,        No --- el     Sí, mismo orden
  (con        cerradas   complementario 26,32 %       árbol, la     
  visión)     citadas    correctamente derivado de    colina, las
                         1 - ratio_pixeles_cromaticos nubes y las
                                                      hierbas se    
                                                      describen     
                                                      como          
                                                      contenido,    
                                                      nunca como    
                                                      sustituto de  
                                                      una métrica   
                                                      cerrada       
  --------------------------------------------------------------------------------------------------

> **Veredicto.** Las cuatro ejecuciones ---dos imágenes en las dos ramas---
> superaron las cuatro preguntas. Ninguna cita el valor de una métrica cerrada
> ---ni siquiera imagen30,
> que con 7 de 10 métricas cerradas es el escenario donde sería más probable
> introducir una cifra no respaldada. Todas las cifras que aparecen en prosa se recomponen
> desde el informe correspondiente mediante las transformaciones pactadas: el
> redondeo correcto a 4 decimales, no el truncado, y el
> complementario 1-v. La única observación que merece una frase en la
> discusión del apartado 9.4 es la de `imagen4` con lectura visual
> («fondo de color claro»): no llega a violar la
> jurisdicción porque no cita ni contradice esquema_cromatico, pero es
> el tipo de frase a vigilar si en el futuro se amplía el corpus de
> ejecuciones con visión.

### Comparación con y sin lectura visual

> Tres de las cinco imágenes representan perfiles de cierre distintos,
> desde una sola condición abierta hasta cuatro de cinco, como muestra
> la Tabla 9.5:

**Tabla 9.5.** Estado (abierta o cerrada) de las cinco condiciones de
aplicabilidad en `imagen30`, `imagen4` e `imagen29`.

  --------------------------------------------------------------
  Condición de aplicabilidad Imagen30    Imagen4     Imagen29
  -------------------------- ----------- ----------- -----------
  Composición                CERRADA     ABIERTA     ABIERTA

  Horizonte                  CERRADA     CERRADA     ABIERTA

  Convergencia               CERRADA     CERRADA     CERRADA

  Espacio y                  CERRADA     ABIERTA     ABIERTA
  aislamiento del sujeto

  Cromática                  ABIERTA     CERRADA     ABIERTA

  Total abiertas             1 de 5      2 de 5      4 de 5
  --------------------------------------------------------------

> **Disciplina de cierre.** Se sostiene sin excepción: ninguna de las diez
> síntesis (5 imágenes × 2 ramas) cita una métrica en metricas_cerradas. Ver la
> fotografía no dio al modelo una excusa para reabrir una condición del nivel 2.
> En los cinco casos,
> la rama con lectura visual describe lo que la condición de aplicabilidad cerró en prosa cualitativa ---«la
> imagen es enteramente acromática», «la posición de este árbol no se
> ajusta a patrones compositivos canónicos»--- sin convertir eso en una
> cifra que el informe no respalda.
>
> La hipótesis era que el contenido visual reduciría la cobertura al añadir frases
> descriptivas sin cifra. El dato apunta en esa dirección en dos de los cinco casos
> (`imagen_puente`, `imagen29`) pero no en los otros tres, y la diferencia
> media entre ramas ---0,03 sobre la fracción de frases citadas que
> define el apartado 8.3.1--- es pequeña frente al ruido de fondo: los cuatro
> diagnósticos de especialista cambian de redacción en cada reejecución
> (verificado --- el informe numérico es idéntico pero el diagnostico en
> prosa nunca lo es, ni siquiera cuando el indicador booleano no toca esa tarea), así
> que parte de la variación observada en la síntesis del crítico no es
> atribuible al canal visual sino a la variabilidad ordinaria del
> muestreo del LLM en las cinco piezas de prosa que la preceden. Con n=5
> no se puede aislar el efecto con más precisión sin una muestra mayor.
>
> El efecto limpio y consistente es el contenido nombrado. La cobertura de
> trazabilidad, medida por frase, no captura la diferencia
> que distingue las dos ramas: en la rama sin lectura visual,
> `observaciones_visuales`
> está vacío en las cinco ejecuciones por construcción, y la síntesis se
> queda en sustantivos genéricos («el sujeto», «una figura»). En la rama
> con lectura visual aparece el contenido identificado de la Tabla 9.6,
> que ninguna métrica
> podría producir:

**Tabla 9.6.** Contenido de la síntesis del crítico, con y sin lectura
visual, por imagen.

  ------------------------------------------------------------
  Imagen             Sin lectura visual   Con lectura visual
                     (síntesis)           (observación visual)
  ------------------ -------------------- --------------------
  Imagen4            «el sujeto», «el     «Un hombre con
                     hombre retratado»    cabello oscuro y
                                          ondulado mira
                                          directamente al
                                           frente […] Su
                                          mano derecha está
                                          apoyada bajo su
                                           barbilla […]
                                           una leve sonrisa.»

  Imagen30           «un sujeto           «Un gran árbol
                     definido», nunca     solitario se alza
                     nombrado             sobre una colina
                                          cubierta de
                                           hierba.»

  Imagen29           «el perro            «Un perro de raza
                     protagonista»        golden retriever
                                          está sentado entre
                                           hierba alta […]
                                           Detrás […] un
                                          cuerpo de agua
                                           tranquilo […] y
                                          una línea de
                                           árboles.»
  ------------------------------------------------------------

> En esos tres perfiles, la tensión fue idéntica entre ramas para la
> imagen de animal, inexistente para el paisaje y distinta para el
> retrato pese a partir de los mismos datos. Con solo estos casos no se
> puede atribuir la diferencia al canal visual; queda como hipótesis que
> una mayor disponibilidad de métricas reduce el margen de variación al
> construir tensiones. En la imagen de animal ocurrió además que la rama
> sin lectura visual dejó sin citar una frase sobre la paleta, mientras
> la rama con lectura visual citó el campo cromático correspondiente. El
> caso aislado no indica que ver la imagen reduzca la disciplina de cita.
>
> El defecto de formular ausencias como «la escena carece de…» apareció
> en las dos ramas del paisaje y en la rama sin lectura visual del
> retrato, pero no en el animal.
> [VERIFICAR-AUTOR: de imagen4 solo se conserva la ejecución con lectura
> visual, y es la que contiene «la escena carece de una estructura
> lineal marcada», de modo que la atribución de este defecto a la rama
> SIN lectura visual del retrato no se puede comprobar con los ficheros
> archivados. Confirmar de memoria o suprimir la mención al retrato.]
> Por tanto, tampoco se atribuye a la
> lectura visual: el patrón es compatible con la dificultad de redactar
> varias condiciones cerradas a la vez.
>
> La lectura visual aporta el sustantivo y la métrica aporta el número,
> sin que uno reemplace al otro y sin coste medible
> en disciplina de cierre.
>
> **Comparación con un modelo generalista.** Queda fuera del alcance: no se ha
> ejecutado ninguna crítica de
> un modelo generalista con la imagen y una instrucción independiente, al margen del
> multiagente, y esa comparación queda declarada como línea de trabajo
> futuro en el apartado 10.4. El argumento de este apartado no es «el sistema
> supera a un modelo de lenguaje aislado», sino
> «la arquitectura de tres fuentes cumple lo que prometía»: el crítico
> incorpora un canal visual sin introducir un coste medible en disciplina de cierre ni en fidelidad
> numérica, y lo que ese canal aporta es precisamente lo que ninguna
> métrica puede dar.
>
> Una segunda acotación, esta de reproducibilidad. De las diez
> ejecuciones comparadas solo se conservan cinco, una rama por imagen,
> porque `outputs/ejecucion/` guarda un único directorio por fotografía y
> cada reejecución sobrescribe la anterior. El experimento es, por
> tanto, revisable en sus conclusiones ---las síntesis conservadas y las
> observaciones citadas en este apartado están en disco--- pero no
> repetible desde los ficheros archivados, a diferencia de la
> comparación de fidelidad del apartado 9.2.2. Archivar las diez
> ejecuciones en un directorio propio, como se hizo con la validación
> del apartado 8.1, es la corrección que evitaría este límite.

## Coste, latencia y consumo de recursos

> **Latencia.** El arranque de la interfaz ---desde `streamlit run app.py` hasta
> que la pantalla de subida está lista---: unos 16 s, dominado por la carga
> a memoria de ResNet18 y YOLO11m (los dos modelos se instancian a nivel
> de módulo, una sola vez por arranque, no por análisis). Un análisis
> completo ---las seis tareas del flujo, desde que se pulsa «Analizar»
> hasta que llega `CriticaCompositiva`--- tarda 2 min 54 s de media sobre
> las mediciones tomadas en la propia interfaz; no es una ejecución única,
> y el valor depende además de la latencia que tenga la interfaz de
> programación en cada momento. Es el coste de encadenar seis llamadas LLM
> secuenciales (`Process.sequential`, sin paralelismo entre especialistas)
> más el I/O de las ocho imágenes de verificación que se escriben a
> disco.
>
> **Coste de la API.** Precio vigente verificado en la documentación oficial
> (ai.google.dev/gemini-api/docs/pricing, consultado el 2026-09-03):
> Flash $0,30 / $2,50 por millón de tokens (entrada/salida); Pro
> $1,25 / $10,00 por millón (contexto ≤200k, que es siempre el caso
> aquí). El orquestador y los cuatro especialistas corren en Flash
> (`.env`: `MODEL=gemini/gemini-2.5-flash`); el crítico en Pro
> (`MODELO_CRITICO` en `crew.py`); con `FLAG_CRITICO_VE_IMAGEN=True` se añade
> una sexta llamada a Flash desde `LecturaVisualTool` --- hasta 7 llamadas
> por análisis.
>
> El coste real, tomado directamente del panel de facturación de Google
> AI Studio para la clave del proyecto, sobre el periodo 6 de junio -- 3
> de septiembre de 2026 (casi tres meses de desarrollo activo), es el
> de la Tabla 9.7:

**Tabla 9.7.** Coste acumulado de la API de Gemini por modelo, sobre
casi tres meses de desarrollo activo del proyecto.

  -----------------------------------------------------------
  Modelo                        Coste acumulado (€)
  ----------------------------- -----------------------------
  Gemini 2.5 Flash              8,76

  Gemini 2.5 Pro                5,44

  Total                         14,20
  -----------------------------------------------------------

> Esta cifra no es el coste marginal de un análisis de producción
> limpio: cubre todo el desarrollo del proyecto --- las sondas de calibración, las
> validaciones parciales de dos tareas por especialista, las iteraciones
> de plantilla de instrucciones que costaron dos y tres rondas por agente, y las
> reejecuciones de calibración documentadas a lo largo de meses.
>
> Los datos corresponden al momento de realizar el proyecto y la
> memoria; el modelo puede cambiar, debido a que puede ser retirado y
> sustituido por modelos más recientes.
>
> **Consumo de recursos.**
>
> **En disco.** Los dos modelos locales pesan 82 MB (39 MB `YOLO11m` + 43 MB
> `ResNet18` con ajuste fino); `outputs/` acumula 572 MB tras las 40 imágenes
> de calibración y las ejecuciones completas conservadas en el momento
> de la medición ---no es una cifra estable, porque las carpetas de
> ejecución crecen y se limpian a lo largo del desarrollo---; y el
> entorno virtual de Python con todas las dependencias instaladas,
> PyTorch y sus binarios de CUDA incluidos, ocupa 5,6 GB.
>
> En GPU (`RTX 4060 Laptop`, 8 GB de `VRAM`): medido con el contador nativo
> de memoria de GPU dedicada por proceso de Windows ---nvidia-smi no
> expone esta cifra por proceso bajo el driver WDDM, así que hizo falta
> esa vía---, muestreando una vez por segundo durante un análisis
> completo real, con los valores que recoge la Tabla 9.8:

**Tabla 9.8.** Consumo de VRAM en reposo, en el pico y su incremento,
durante un análisis completo.

  -----------------------------------------------------------
                               VRAM
  ---------------------------- ------------------------------
  En reposo con los modelos ya 198,8 MiB
  cargados                     

  Pico durante el análisis     532,8 MiB

  Incremento                   334,0 MiB aproximadamente un
                               4 % de los 8 GB disponibles
  -----------------------------------------------------------

> El incremento es pequeño y consistente con la arquitectura híbrida
> declarada (capítulo 1): solo ResNet18 y YOLO11m corren en GPU local; las siete
> llamadas a Gemini se ejecutan en la nube y no consumen VRAM del
> equipo. Es un dato que refuerza el argumento de que el sistema puede
> correr en un equipo de consumo, no en una estación dedicada.

## Qué demuestran estos resultados y qué no

> Este apartado interpreta los resultados de los apartados 9.1 a 9.3.
>
> **Lo que sí demuestra.**

- Que el contrato de confianza por métrica funciona en la práctica. Sobre
  `imagen18`, los siete
  incumplimientos que la revisión encontró pasaron a ninguno al corregir el
  defecto de plantilla de instrucciones que los causaba, con los mismos
  informes numéricos de entrada y las dos ejecuciones archivadas --- citar el
  valor de una métrica con confianza 0 era un fallo real y resultó ser
  corregible. La revisión visual panel a panel del apartado 9.2.1 acompaña ese
  resultado, aunque mide consistencia interna y no generalización porque los
  umbrales se calibraron sobre ese mismo corpus y con esa misma revisión.

- Que el sistema no rellena con prosa lo que no puede medir: sobre las cuatro
  ejecuciones revisadas frase a frase, toda cifra del texto se recompone desde
  su informe y ninguna procede de una métrica cerrada (apartado 9.2.2).

- Que la rama con lectura visual (crítico con imagen) no reabre condiciones de aplicabilidad cerradas ni siquiera
  en el caso más exigente de los comparados (`imagen30`, con 7 de sus 10
  métricas cerradas y una sola condición abierta de cinco), donde la
  tentación de rellenar con observaciones visuales es máxima. La comparación
  sistemática entre las dos ramas, con su tabla sobre tres perfiles de
  cierre distintos, está en el apartado 9.2.3.

- Que la fidelidad de transcripción es exacta en los casos verificados: el
  100 % de los campos del informe en las dos validaciones de extremo a extremo
  del Agente 2 (apartado 6.3.5) y las 8 de 8 cifras de cada una de las cuatro
  ejecuciones revisadas a mano (apartado 9.2.2), correctamente redondeadas y
  no truncadas.

> **Lo que no demuestra.**

- No demuestra que el sistema evalúe «mejor» la composición que un
  humano o que un LLM generalista.

- No demuestra la validez externa de los umbrales, desarrollada en la limitación
  i del apartado 9.5.

- No demuestra cómo se comporta la cobertura de trazabilidad entre las dos
  ramas del crítico: con cinco pares de ejecuciones, la diferencia media
  ---0,03--- no se distingue de la variabilidad ordinaria de la prosa
  generada (apartado 9.2.3).

## Limitaciones

> Las principales limitaciones transversales del sistema son las
> siguientes:

i.  Validez externa de los umbrales --- calibrados sobre las 40 imágenes
    propias del corpus de desarrollo. `RATIO_CROMATICO_MIN=0.10` pasó de
    tener el margen más ancho del sistema a abrirse por solo 0,0029 con
    `imagen39`; `LONGITUD_MIN_HORIZONTE=0.13` separa por 0,0116.

ii. Clases ausentes de COCO --- YOLO no reconoce árbol, edificio,
    montaña ni terreno como sujeto, así que la condición de aplicabilidad del Agente 1 y la
    del Agente 3 cierran siempre ahí aunque el sujeto sea real y claro
    (`imagen13`, `imagen30`). Aceptado explícitamente porque cae justo donde W
    pesa menos (arquitectura, 0,17).

iii. `ratio_nitidez` mide densidad de detalle, no enfoque --- confirmado
     en `imagen27`/`imagen33`: el signo acierta pero la magnitud exagera lo
     que percibe un humano (la varianza del Laplaciano no es
     perceptualmente lineal).

iv. Resolución angular ~1° en `angulo_horizonte` (cuantización de `theta`
    en Hough) --- 14 de las 40 imágenes del corpus de desarrollo dan
    exactamente 0,00°.

v.  `cv2.grabCut` es determinista pero no estable: un desplazamiento
    sub-píxel del `bbox` mueve la cobertura decenas de puntos. La
    inestabilidad se concentra en los casos que la condición de
    aplicabilidad ya cierra, lo que apunta a favor de esa condición; con
    una sola observación y sin un experimento de perturbación controlada,
    no pasa de ser un indicio.

vi. W es hipótesis de diseño, no aprendida --- no hay validación
    estadística de que esos pesos sean los óptimos. Los valores se
    fijaron mediante razonamiento bibliográfico fila a fila (Freeman,
    Präkel y las correlaciones de EVA), con un sondeo de percepción y
    las propuestas de tres modelos de lenguaje como referencias previas
    (apartado 4.6.2); ninguna de las dos es evidencia independiente, y
    el sondeo discrepa de la matriz final en el contexto de producto.

vii. Validación autoevaluada y no ciega --- la revisión de las condiciones
     de aplicabilidad (apartado 9.2.1) y la de fidelidad métrica-texto
     (apartado 9.2.2) las realiza el propio autor, sin evaluador externo,
     sin segundo anotador y sobre el mismo corpus con el que se calibraron
     los umbrales que se juzgan. Es una diferencia de naturaleza respecto
     de la cifra del apartado 9.1, obtenida sobre una partición reservada
     del conjunto de datos. La evaluación ciega con un panel de fotógrafos
     que la sustituiría se declara como línea de trabajo futuro en el
     apartado 10.4.

# Conclusiones y trabajo futuro

Este capítulo cierra el trabajo con el cumplimiento de los objetivos, las
aportaciones surgidas durante el desarrollo, las lecciones aprendidas y las líneas
de trabajo futuro. Cada afirmación remite a la cifra o el artefacto que la sostiene;
cuando no hay evidencia, se declara.

## Grado de cumplimiento de los objetivos

> El objetivo general ---desarrollar una aplicación en Python que realice un
> análisis compositivo de fotografías digitales mediante un sistema multiagente
> construido sobre CrewAI--- se ha alcanzado: el sistema está implementado, se
> ejecuta de principio a fin sobre una fotografía cualquiera y entrega una crítica
> citada junto con su evidencia visual. A continuación se revisan los cuatro
> objetivos específicos, su evidencia y la parte que quedó fuera.
>
> **Objetivo 1 --- arquitectura jerárquica con un agente orquestador que coordine
> a los especialistas de cada dimensión compositiva.** Se entrega en tres niveles,
> seis agentes y seis tareas, con el reparto que describen los capítulos 5, 6 y 7 y
> con los contratos de datos y el aislamiento de contexto del apartado 4.3. Dos
> evidencias confirman que la jerarquía opera. Primero, un análisis completo es
> las seis tareas encadenadas y tarda algo menos de tres minutos, con hasta siete
> llamadas a la interfaz de programación (apartado 9.3). Segundo, el orden
> que impone el vector de pesos se cumple en la salida: en las cuatro ejecuciones
> revisadas del apartado 9.2.2, el discurso de la síntesis sigue el
> `orden_dimensiones` que fijó el arbitraje, y no otro. A ello se añade la
> comprobación del apartado 7.2.5, donde privar al crítico de la herramienta que le
> suministra el vector le llevó a inventarse una fila de pesos plausible pero
> incorrecta: el aislamiento arquitectónico no era una cautela sobrante.
> **Lo que queda fuera**: la coordinación no es de
> mando en tiempo de ejecución. Los cuatro especialistas se ejecutan siempre, bajo
> `Process.sequential`, y ningún modelo de lenguaje decide qué agente interviene; es
> una jerarquía de responsabilidad, y el motivo de haberla elegido así ---que una
> activación decidida por un modelo no sería auditable--- está en el apartado 2.6.
>
> **Objetivo 2 --- integrar YOLO y algoritmos clásicos de visión por computador,
> en lugar de los modelos de segmentación y de profundidad que contemplaba el
> anteproyecto, para extraer métricas cuantitativas verificables.** Se entregan
> diez métricas repartidas en cuatro especialistas, calculadas con detección de
> objetos, saliencia por residuo espectral, Canny y Hough con estimación del punto
> de fuga por consenso, GrabCut y varianza del laplaciano, y análisis cromático en
> CIELAB y HSV (capítulos 5 y 6). Que calculan lo que dicen calcular lo sostienen
> las nueve pruebas analíticas con valor de referencia conocido, cuyo método
> describe el apartado 8.3.2 y cuyo alcance recoge el apartado 8.1, entre ellas la
> del punto de fuga trazado a mano en (0,6836; 0,3660) y recuperado por el
> estimador en (0,6812; 0,3662). Que lo calculado es
> además compositivamente defendible lo sostiene la revisión visual panel a panel
> del apartado 9.2.1, con tasas de acierto del 100 %, 87,2 %, 76,9 % y 94,9 % para
> los cuatro especialistas sobre las 39 imágenes del corpus de aplicabilidad.
> **Lo que queda fuera**: SAM y MiDaS no se han utilizado, por los motivos que
> desarrolla el apartado 6.4.3, y con MiDaS decae la jerarquía de planos que el
> anteproyecto asociaba a esta dimensión. Y las métricas son verificables y
> recomputables, que es lo que el objetivo pedía, pero no están validadas contra
> una referencia externa anotada: no se ha medido cuánto se equivoca el punto de
> fuga cuando afirma, porque eso exige un corpus con la perspectiva etiquetada que
> este trabajo no tiene (apartado 10.4).
>
> **Objetivo 3 --- informes basados en datos cuantificables y un agente crítico
> que los sintetice priorizando las dimensiones más relevantes de cada contexto
> según el vector de pesos.** Se entrega con la doble autoría del apartado 4.5
> ---un informe que produce el motor de cálculo y es recomputable, y un diagnóstico
> que redacta el modelo de lenguaje citando ese informe--- y con el arbitraje
> determinista del apartado 7.1, que aplica el vector sin que ningún modelo de
> lenguaje intervenga. La evidencia está en los apartados 9.2.2 y 9.2.3: las cuatro
> preguntas de la revisión manual se superan en las cuatro ejecuciones examinadas
> ---ninguna cifra del texto sin respaldo en su informe, y correctamente redondeada
> en lugar de truncada---, y sobre las diez síntesis del experimento de ramas
> ninguna cita el valor de una métrica que el propio sistema declaró no aplicable,
> tampoco cuando el crítico dispone de una lectura visual de la fotografía.
> **Lo que queda fuera**: el mecanismo de priorización se cumple, pero los pesos
> concretos no están validados. W es una hipótesis de diseño justificada fila a
> fila con la literatura fotográfica y las correlaciones del conjunto de datos EVA,
> no un parámetro aprendido ni optimizado contra ninguna métrica, y así se recoge
> en la sexta limitación del apartado 9.5.
>
> **Objetivo 4 --- interfaz gráfica que permita interactuar con el sistema y
> muestre junto a la crítica la evidencia visual que la sostiene.** Se entrega la
> aplicación descrita en el apartado 7.6: se sube una fotografía, se sigue el
> progreso de las seis tareas y se recibe la síntesis, las salvedades y seis
> paneles de verificación, con una leyenda que explica en qué escala vive cada
> métrica citada. La evidencia de que el circuito completo funciona por esa vía es
> la medición de latencia del apartado 9.3, tomada en la propia interfaz: unos 16
> segundos de arranque y unos 2 minutos y 54 segundos por análisis.
> **Lo que queda fuera**: la usabilidad no se ha
> evaluado con ningún instrumento. No hay prueba con usuarios ni protocolo que la
> mida en el capítulo 8, de modo que lo que este trabajo puede afirmar es que la
> interfaz existe, entrega la crítica con su evidencia y se ha usado para medir el
> sistema, no que se haya demostrado que un fotógrafo de cualquier nivel la maneje
> de forma intuitiva. Tampoco hay despliegue público, por las razones que declara
> el apartado 1.5.

## Aportaciones del trabajo

> Las aportaciones siguientes no estaban comprometidas en el anteproyecto y
> surgieron de la propia disciplina de calibración. Se presentan cinco,
> ordenadas de más a menos arquitectónicas:
>
> **Contrato de confianza por métrica.** El contrato `MetricaConfianza`
> (`valor` + `confianza` + `fuente`) convierte «esta métrica no aplica aquí» de un
> flujo de control ad-hoc (saltarse agentes) en un dato que cada una de
> las diez métricas autodeclara. Es lo que permite que el sistema
> prefiera callar a fabricar falsa precisión, y es la pieza que ningún
> documento previo pedía.
>
> **Separación de autorías entre informe y diagnóstico.** Produce dos fidelidades
> distintas (transcripción y métrica-texto) en vez de una sola caja negra LLM, y
> convierte cada salida en algo recomputable por un tercero.
>
> **Aislamiento arquitectónico del vector W.** El aislamiento no depende solo de una
> instrucción, sino también de la estructura del grafo (W no está en la salida del
> orquestador, luego no puede llegar a los especialistas). Tiene
> evidencia empírica a favor: privado de la herramienta que se lo
> suministra, el crítico se inventó la fila de W en lugar de negarse a
> responder (apartado 7.2.5). Esta observación respalda que el aislamiento
> no era una cautela sobrante.
>
> **Cobertura de trazabilidad como métrica de validación nueva.** Esta métrica,
> definida en el apartado 8.3.1, cuantifica qué fracción de una crítica
> descansa en algo verificable
> frente a algo que no lo es. No es una cifra que el anteproyecto
> pidiera, y es la que hace defendible haber ampliado el crítico con un
> canal visual (rama con lectura visual) sin que eso se convierta en una caja negra más.
> La conclusión que deja esa ampliación es que el canal visual no se paga
> donde se temía: en la imagen del corpus con más métricas cerradas ---siete
> de diez, y una sola condición de aplicabilidad abierta de cinco---, que es
> donde rellenar con lo que se ve resultaría más fácil, el crítico que dispone
> de la fotografía describe en prosa cualitativa aquello que el nivel 2 cerró
> y no lo convierte en ninguna cifra. Nombra lo que la métrica no puede
> nombrar, y no toca lo que la métrica sí mide (apartado 9.2.3).
>
> **Metodología de calibración reproducible y hallazgo transversal.** El
> ciclo contrato provisional → sonda de calibración → revisión visual → contrato
> cerrado, aplicado de forma consistente a las diez métricas, con el
> criterio explícito de banda vacía y la asimetría
> falso-positivo/falso-negativo. De ahí sale un hallazgo que vale para
> más que este proyecto: tres algoritmos distintos (*k-means*, RANSAC,
> GrabCut) resultaron no deterministas por tres causas completamente
> distintas (orden de reducción en coma flotante, semilla de muestreo,
> RNG interno sin API expuesta) --- nada de esto era anticipable desde
> el anteproyecto y solo apareció al comprobarlo explícitamente.

## Lecciones aprendidas

> Las lecciones siguientes conectan con las aportaciones del apartado 10.2:
> son hallazgos que no cambiaron ninguna línea de funcionalidad, pero que se
> perderían si no se documentaran. Constituyen el tipo de contenido que
> distingue una memoria de una documentación técnica. Se presentan seis,
> ordenadas de la más repetida a la más puntual:
>
> «Una prohibición no gana a una instrucción que pide lo contrario», y
> el arreglo es borrar, no añadir. Es el hallazgo más reincidente del
> proyecto --- apareció en los Agentes 1, 2, 3, 4 y en el crítico (al
> menos seis veces documentadas). El patrón siempre fue el mismo:
> el campo `backstory` prohibía citar una métrica cerrada, y el campo
> `expected_output` la pedía igualmente («una confianza de 0 no te
> autoriza a callarte la métrica: la mencionas»). El LLM obedecía la instrucción concreta
> sobre la prohibición general. La solución que funcionó las seis veces
> no fue añadir una prohibición más larga, sino bifurcar el propio
> `expected_output` para que los puntos que citan una métrica cerrada
> dejen de existir cuando su confianza es 0. Lección transferible a
> cualquier sistema con LLM: una plantilla de instrucciones larga acumula contradicciones
> internas antes que faltas de especificación, y hay que auditarla por
> consistencia, no solo por cobertura.
>
> El no determinismo es invisible si no se busca a propósito, y su causa
> cambia con el algoritmo. Se encontró en tres piezas distintas por tres
> motivos distintos: *k-means* con `random_state` fijo seguía sin ser
> reproducible por el orden de reducción en coma flotante de
> `BLAS/OpenMP`; `RANSAC` con muestreo aleatorio hace depender la
> puntuación según la semilla elegida (resuelto mediante enumeración
> exhaustiva, viable porque la
> muestra mínima son pares); `cv2.grabCut` no expone ninguna semilla
> propia y depende del RNG global de OpenCV (`cv2.setRNGSeed()` antes de
> cada llamada). Ninguno de los tres se habría detectado sin comprobar
> explícitamente «¿esto da lo mismo si lo repito?» --- la lección de
> método es que en un sistema que promete verificabilidad, el
> determinismo no se puede asumir de la documentación de una librería,
> hay que medirlo.
>
> Un fallo de herramienta puede reportarse como éxito, y es la clase de
> error más peligrosa en un sistema agéntico. `gemini-2.5-flash` se
> quedaba mudo (Invalid response from LLM call --- None or empty) al
> combinar una plantilla de instrucciones larga con llamadas nativas a
> herramientas, y solo se detectó
> aislando variables una a una hasta encontrar la combinación exacta que
> rompía. El registro de cambios de CrewAI (versión 1.15.9) recoge un fallo de
> la misma familia ---los fallos de herramienta se reportaban como éxito---, lo
> que sitúa lo observado dentro de un problema conocido del ecosistema y no como
> una rareza de este proyecto, aunque no sea exactamente el mismo caso. La lección: en un sistema donde el LLM
> puede rellenar en prosa el hueco que deja una herramienta que falló en
> silencio, cada tarea nueva que crece en contexto necesita una
> comprobación explícita de que su herramienta sigue ejecutándose --- el
> síntoma de un fallo así no es un error legible.
>
> Una validación de extremo a extremo demuestra que una plantilla de instrucciones funcionó una
> vez, no que sea robusta. Al actualizar CrewAI, con los informes de
> entrada idénticos, tres de los cuatro diagnósticos reintrodujeron
> defectos ya corregidos meses antes --- no por la actualización, sino
> por variabilidad del LLM entre ejecuciones con la misma plantilla de instrucciones. La
> solución fue sustituir prohibiciones dispersas por una comprobación
> final idéntica en los cuatro agentes, un paso explícito de relectura
> con permiso para borrar frases. Consecuencia metodológica: cada
> validación de este trabajo debería leerse como «funcionó en esta
> ejecución», y el instrumento de defensa frente a eso es un
> procedimiento de revisión repetible, no una ejecución afortunada.
>
> La revisión visual humana encontró sistemáticamente lo que el diseño
> no anticipó --- y una condición de aplicabilidad que deja pasar casi todo el corpus es ella
> misma la señal de que algo falla. Ocurrió con la condición de aplicabilidad del horizonte
> (dejaba pasar al 90 %), la de convergencia (dos condiciones no
> bastaban), la del Agente 1 (97,5 %) y la del Agente 3 (donde las dos
> condiciones previstas se descartaron por completo tras mirar las
> imágenes). En los cuatro casos la corrección no vino de más teoría
> sino de mirar la salida. Lección de método: una métrica automática de
> calibración (banda vacía, percentil) es necesaria pero no suficiente;
> hace falta una revisión humana dirigida como paso del propio proceso
> de desarrollo, no como ocurrencia tardía.
>
> Un umbral situado en el borde de una banda vacía tiene menos margen que uno
> centrado, aunque ambos funcionen el día de la calibración. El caso de
> `RATIO_CROMATICO_MIN=0.10` se cuantifica en el apartado 6.5.4: la banda seguía
> siendo ancha, pero el corte estaba mal situado dentro de ella. Es una lección de
> calibración generalizable más allá de este proyecto: la anchura de la
> banda vacía es una condición necesaria para defender un umbral, pero
> no dice nada sobre su margen de seguridad frente a datos futuros si el
> corte no está centrado en ella.

## Líneas de trabajo futuro

> Las líneas siguientes responden a varias limitaciones del apartado 9.5
> y añaden extensiones que quedaron fuera del alcance del trabajo. Se
> organizan en cuatro bloques:
>
> **Arquitectura y despliegue.**
>
> Se propone sustituir la API de Gemini por un modelo de lenguaje local
> para el crítico. La propia premisa arquitectónica del sistema lo motiva:
> todo lo demás corre en local sobre GPU y el crítico es la única pieza en
> la nube, con su coste, su latencia y su dependencia de conexión
> asociados.
>
> También se propone un visor de solo lectura que muestre análisis ya
> calculados desde `outputs/ejecucion/{stem}/`, sin ejecutar el flujo de
> procesamiento, `torch` ni `YOLO`, y sin exponer la clave de la API. Esta
> sería la vía que permitiría un enlace público sin invalidar la validación
> medida sobre el modelo YOLO actual.
>
> **Calibración y normalización.** Este bloque enlaza directamente con la
> primera limitación del apartado 9.5: la validez externa de los umbrales.
>
> Se propone normalizar por percentiles en vez de por cota teórica. Esta
> opción está pendiente para `d_equilibrio` (emite `valor_norm=None` desde
> el diseño porque su cota teórica comprimiría el corpus en [0,68–1,00])
> y es candidata para `longitud_horizonte` y `margen_patron`, que no tienen
> el ancla de dominio absoluta que sí se logró para el Agente 4.
>
> También se propone recalibrar la condición de aplicabilidad de
> convergencia perspectiva tras el primer falso positivo confirmado
> (`imagen32`, producto con apertura 89,3°). Con un solo caso no hay corpus
> para decidir si hace falta una cuarta condición, pero se trata de una
> alerta abierta que el corpus creciente debe resolver.
>
> **Validación empírica del vector de pesos contextuales.** La sexta
> limitación del apartado 9.5 declara que W es una hipótesis de diseño y
> que no hay evidencia estadística de que esos valores sean los
> adecuados. Medir su efecto es barato, porque el vector vive aislado en
> un fichero de parámetros y perturbar una fila no exige tocar código:
> basta comprobar si cambia el orden del discurso y la crítica
> resultante. Ajustarlo contra juicios humanos anotados es el paso
> siguiente, y ese sí exige el panel del bloque de validación externa.
>
> Se propone comprimir la escala de `ratio_nitidez` mediante una lectura
> logarítmica del cociente, para que el lenguaje del diagnóstico se
> corresponda mejor con lo que percibe un observador, dado que la
> varianza del Laplaciano no es perceptualmente lineal (confirmado en
> `imagen27` y `imagen33`).
>
> **Validación externa.** Este bloque enlaza con las exclusiones ya
> declaradas en el apartado 9.5.
>
> Se propone utilizar un corpus anotado externo, como York Urban Database,
> para medir la precisión del punto de fuga contra una referencia anotada.
> Hoy la métrica declara cuándo no es aplicable, lo que impide que afirme
> una perspectiva inexistente, pero no se ha medido cuánto se equivoca
> cuando sí afirma: eso exige imágenes con el punto de fuga anotado, que
> este corpus propio no tiene.
>
> También se propone evaluar a ciegas con un panel de fotógrafos para sustituir
> la revisión autoevaluada del propio autor que
> sostiene la validación actual, y una prueba de usabilidad de la interfaz
> con usuarios de distinto nivel. Este es el instrumento que hoy falta para
> sostener por completo el cuarto objetivo específico, según acota el
> apartado 10.1.
>
> **Ablación más amplia del sistema.** El trabajo actual compara las dos
> configuraciones que el sistema admite hoy —con y sin lectura visual—.
> Quedan sin medir otras dos, que exigen un diseño distinto:
> un modelo generalista con la imagen y una instrucción independiente, al margen del
> sistema, y el sistema completo pero con los especialistas emitiendo
> solo métricas, sin sus diagnósticos en prosa. La primera es la
> comparación que el apartado 9.2.3 declara fuera de alcance.
>
> **Ampliación del sistema.**
>
> Se propone estudiar el veto asimétrico del canal visual: permitirle
> rebajar —nunca elevar— una afirmación métrica según lo que ve, en lugar
> de limitarse a añadir observaciones marcadas como hace hoy. Esta extensión
> quedó fuera del producto mínimo viable a propósito y por los motivos que detalla el
> apartado 7.4.6.
>
> Por último, se propone estudiar si la concentración de saliencia puede
> emplearse como puntuación continua para graduar `sujeto_discreto`. Su uso
> como umbral binario se descartó porque los valores medidos no formaban
> grupos separables; el posible uso continuo no se evaluó.
>
> Ninguna de estas líneas cambia lo que el trabajo defiende. El sistema
> entregado no puntúa mejor que los que ya existen: se deja interrogar.
> Cada frase de su crítica remite a una magnitud que otra persona puede
> recalcular a partir de la misma fotografía, y cada silencio tiene detrás
> una condición de aplicabilidad declarada y no un olvido. Es la diferencia
> entre un sistema que se cree y uno que se comprueba, y es lo que este
> trabajo ha querido demostrar que se puede construir.

# Bibliografía

[1] L. Marchesotti, N. Murray y F. Perronnin, «Discovering beautiful
attributes for aesthetic image analysis», *International Journal of Computer
Vision*, vol. 113, n.º 3, pp. 246–266, 2015, doi:
10.1007/s11263-014-0789-2.

[2] M. Daryanavard Chounchenani, A. Shahbahrami, R. Hassanpour y G.
Gaydadjiev, «Deep learning based image aesthetic quality assessment—A
review», *ACM Computing Surveys*, vol. 57, n.º 7, art. 183, 2025, doi:
10.1145/3716820.

[3] M. Freeman, *El ojo del fotógrafo: composición y diseño para crear
mejores fotografías digitales*. Barcelona, España: Blume, 2008.

[4] D. Präkel, *Composición*. Barcelona, España: Blume, 2007.

[5] N. Murray, L. Marchesotti y F. Perronnin, «AVA: A large-scale database
for aesthetic visual analysis», en *2012 IEEE Conference on Computer Vision
and Pattern Recognition*, 2012, pp. 2408–2415, doi:
10.1109/CVPR.2012.6247954.

[6] C. Kang, G. Valenzise y F. Dufaux, «EVA: An explainable visual
aesthetics dataset», en *Joint Workshop on Aesthetic and Technical Quality
Assessment of Multimedia and Media Analytics for Societal Trends*, 2020,
doi: 10.1145/3423268.3423590.

[7] J. Canny, «A computational approach to edge detection», *IEEE
Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-8,
n.º 6, pp. 679–698, 1986, doi: 10.1109/TPAMI.1986.4767851.

[8] J. Matas, C. Galambos y J. Kittler, «Robust detection of lines using the
progressive probabilistic Hough transform», *Computer Vision and Image
Understanding*, vol. 78, n.º 1, pp. 119–137, 2000, doi:
10.1006/cviu.1999.0831.

[9] M. A. Fischler y R. C. Bolles, «Random sample consensus: A paradigm
for model fitting with applications to image analysis and automated
cartography», *Communications of the ACM*, vol. 24, n.º 6, pp. 381–395,
1981, doi: 10.1145/358669.358692.

[10] X. Hou y L. Zhang, «Saliency detection: A spectral residual approach»,
en *2007 IEEE Conference on Computer Vision and Pattern Recognition*, 2007,
doi: 10.1109/CVPR.2007.383267.

[11] N. Otsu, «A threshold selection method from gray-level histograms»,
*IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, n.º 1,
pp. 62–66, 1979, doi: 10.1109/TSMC.1979.4310076.

[12] C. Rother, V. Kolmogorov y A. Blake, «“GrabCut”: Interactive
foreground extraction using iterated graph cuts», *ACM Transactions on
Graphics*, vol. 23, n.º 3, pp. 309–314, 2004, doi:
10.1145/1015706.1015720.

[13] A. R. Smith, «Color gamut transform pairs», *ACM SIGGRAPH Computer
Graphics*, vol. 12, n.º 3, pp. 12–19, 1978, doi:
10.1145/965139.807361.

[14] K. He, X. Zhang, S. Ren y J. Sun, «Deep residual learning for image
recognition», en *2016 IEEE Conference on Computer Vision and Pattern
Recognition*, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[15] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh y D.
Batra, «Grad-CAM: Visual explanations from deep networks via gradient-based
localization», en *2017 IEEE International Conference on Computer Vision*,
2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[16] A. Dosovitskiy *et al.*, «An image is worth 16×16 words: Transformers
for image recognition at scale», en *International Conference on Learning
Representations*, 2021, arXiv:2010.11929.

[17] L. Wang *et al.*, «A survey on large language model based autonomous
agents», *Frontiers of Computer Science*, vol. 18, n.º 6, art. 186345,
2024, doi: 10.1007/s11704-024-40231-1.

[18] A. Adimulam, R. Gupta y S. Kumar, «The orchestration of multi-agent
systems: Architectures, protocols, and enterprise adoption», 2026,
arXiv:2601.13671.

[19] H. Talebi y P. Milanfar, «NIMA: Neural image assessment», *IEEE
Transactions on Image Processing*, vol. 27, n.º 8, pp. 3998–4011, 2018,
doi: 10.1109/TIP.2018.2831899.

[20] S. Kong, X. Shen, Z. Lin, R. Mech y C. Fowlkes, «Photo aesthetics
ranking network with attributes and content adaptation», en *Computer
Vision—ECCV 2016*, 2016, pp. 662–679, doi:
10.1007/978-3-319-46448-0_40.

[21] X.-C. Liu y J. Wagemans, «From concepts to judgments: Interpretable
image aesthetic assessment», 2026, arXiv:2603.18108.

[22] D. Qi *et al.*, «The photographer's eye: Teaching multimodal large
language models to see, and critique like photographers», en *2025 IEEE/CVF
Conference on Computer Vision and Pattern Recognition*, 2025,
pp. 24807–24816.

[23] Y. Abe, T. Daikoku y Y. Kuniyoshi, «AI outperforms humans in
personalized image aesthetics assessment via LLM-based interviews and
semantic feature extraction», 2026, arXiv:2605.14761.

[24] W. Liu y Z. Wang, «A database for perceptual evaluation of image
aesthetics», en *2017 IEEE International Conference on Image Processing*,
2017, pp. 1317–1321, doi: 10.1109/ICIP.2017.8296495.
