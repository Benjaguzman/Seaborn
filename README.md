# Seaborn

### Seaborn es una biblioteca de visualización de datos en Python basada en Matplotlib. Proporciona una interfaz de alto nivel para crear gráficos estadísticos atractivos y fáciles de interpretar. Seaborn simplifica la creación de gráficos complejos, como mapas de calor, gráficos de violín y gráficos de dispersión, y permite trabajar fácilmente con datos en formato de DataFrame de pandas. Además, incluye paletas de colores y estilos que ayudan a mejorar la presentación visual de los datos.
![image](https://github.com/user-attachments/assets/b5e2fc56-2937-41c4-a5fd-440d13100868)

## Tipos de Gráficos en Seaborn
### Seaborn ofrece una variedad de gráficos que son ideales para diferentes tipos de visualización de datos. Aquí algunos de los más comunes:

Gráficos de dispersión (scatter plots): sns.scatterplot()
~~~
iris = sns.load_dataset("iris")
sns.relplot(x = "sepal_length", y = "sepal_width", data = iris);
~~~

![image](https://github.com/user-attachments/assets/b17f4045-79bd-4a36-9ddc-52ed8d62cb4e)
~~~
Gráficos de líneas (line plots): sns.lineplot()
df = pd.DataFrame({
    "x": range(100),
    "y": np.random.randn(100).cumsum()
})
sns.relplot(x = "x", y = "y", data = df, kind = "line");
~~~

![image](https://github.com/user-attachments/assets/229f8010-be8d-4f19-8af5-53b8c8f489cd)

Gráficos de barras (bar plots): sns.barplot()
~~~
titanic = sns.load_dataset("titanic")
sns.catplot(x = "sex", y = "survived", kind = "bar", data = titanic);
~~~
![image](https://github.com/user-attachments/assets/42fcdd84-6e9e-4524-94c3-bad74201b5c9)
~~~
Gráficos de cajas (box plots): sns.boxplot()

import seaborn as sns

# Box plot
sns.boxplot(x = variable)

# Equivalente a:
sns.boxplot(x = "variable", data = df)
~~~

![image](https://github.com/user-attachments/assets/e73e5e0f-f956-4d67-8755-678a9c57a06a)

Gráficos de violín (violin plots): sns.violinplot()
~~~
import seaborn as sns

# Gráfico de violín
sns.violinplot(x = variable)

# Equivalent to:
sns.violinplot(x = "variable", data = df)
~~~
![image](https://github.com/user-attachments/assets/bd792c33-f595-49a0-b7ed-192e94f45d69)

Gráficos de pares (pair plots): sns.pairplot()
~~~
import seaborn as sns

sns.pairplot(df, vars = ["petal_length", "petal_width"])
~~~
![image](https://github.com/user-attachments/assets/20ce0818-864c-41fd-8d49-559402acb786)

Mapas de calor (heatmaps): sns.heatmap()
~~~
import numpy as np
import seaborn as sns

# Simulación de datos
np.random.seed(1)
data = np.random.rand(10, 10)

sns.heatmap(data)
~~~

![image](https://github.com/user-attachments/assets/0abbec5d-dec7-44f2-878f-6c92ad64aada)

Gráficos de distribución (distribution plots): sns.histplot(), sns.kdeplot()
~~~
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
data = np.random.randn(1000)

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True, color='blue', stat='density', alpha=0.6)

plt.title('Distribución de Datos con Histograma y KDE')
plt.xlabel('Valor')
plt.ylabel('Densidad')

plt.show()
~~~

![image](https://github.com/user-attachments/assets/01065b16-c1a2-4b35-950f-e9bd0ed8abd0)

Gráficos de regresión (regression plots): sns.regplot()
~~~
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(0)
x = np.random.rand(100) * 10
y = 2 * x + np.random.normal(0, 5, 100)
data = pd.DataFrame({'X': x, 'Y': y})

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.regplot(x='X', y='Y', data=data, marker='o', color='blue', line_kws={'color': 'red'})

plt.title('Gráfico de Regresión Lineal')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')

plt.show()
~~~

![image](https://github.com/user-attachments/assets/4f8881fa-9b84-4043-94a7-0d09caa91850)

# Estilos para una Presentación Más Profesional

## *Seaborn permite personalizar los gráficos para que se vean más atractivos. Aquí hay algunos consejos para lograr un estilo más profesional:*


## Selecciona un estilo: Usa sns.set_style() para elegir entre opciones como 'white', 'dark', 'whitegrid', 'darkgrid', o 'ticks'.

~~~
import seaborn as sns
sns.set_style("whitegrid")
~~~
## Paletas de colores: Utiliza paletas predefinidas o personalizadas con sns.color_palette(). Puedes optar por paletas como 'deep', 'muted', 'pastel', etc.
~~~
sns.set_palette("deep")
~~~
## Tamaños y figuras: Ajusta el tamaño de la figura con plt.figure(figsize=(ancho, alto)) antes de crear el gráfico.
~~~
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
~~~
## Etiquetas y títulos: Añade títulos y etiquetas claras usando plt.title(), plt.xlabel(), y plt.ylabel().
~~~
plt.title("Título del Gráfico")
plt.xlabel("Etiqueta X")
plt.ylabel("Etiqueta Y")
~~~
## Formato de ejes: Personaliza los límites y ticks de los ejes con plt.xlim(), plt.ylim(), y plt.xticks(), plt.yticks().
~~~
plt.xlim(0, 100)  # Limitar eje X
plt.ylim(0, 100)  # Limitar eje Y
~~~
## Anotaciones y Leyendas Usa anotaciones para destacar puntos importantes y agrega leyendas cuando sea necesario:
~~~
plt.legend(title="Leyenda")
~~~
## Guardar Gráficos en Alta Calidad Guarda tus gráficos en formatos de alta calidad para presentaciones
~~~
plt.savefig('nombre_grafico.png', dpi=300)  # Alta resolución
~~~
## Espaciado y Diseño Asegúrate de que los elementos de tus gráficos no se solapen y que haya suficiente espacio:
~~~
plt.tight_layout()  # Mejora el espaciado
~~~

Guardar gráficos: Guarda tus gráficos en formatos de alta calidad usando plt.savefig('nombre.png', dpi=300), lo que es ideal para presentaciones.



