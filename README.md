# Introdução aos Tensores com PyTorch 2.0

---

## 1. Introdução 📖

<p style='text-align: justify;'>Bem-vindo a este Jupyter Notebook, onde exploraremos o conceito fundamental do PyTorch 2.0 - os tensores.</p>

O <a href="https://pytorch.org/get-started/pytorch-2.0/" target="_blank">PyTorch</a> é uma biblioteca de aprendizado de máquina de código aberto para Python, desenvolvida principalmente pela equipe de pesquisa de inteligência artificial do <a href="https://web.facebook.com/?_rdc=1&_rdr" target="_blank">Facebook</a>. É usado para aplicações como processamento de linguagem natural e foi projetado para permitir a computação eficiente de tensores com aceleração de GPU.

Um tensor é uma generalização de vetores e matrizes para um número maior de dimensões e é uma entidade matemática muito importante no aprendizado profundo. No PyTorch, tudo é um tensor. Seja uma imagem, um vetor de recursos, um conjunto de parâmetros de modelo, todos são representados como tensores.

Neste notebook, vamos mergulhar fundo no mundo dos tensores. Vamos começar com a criação de tensores, entender suas propriedades e operações, e ver como eles são usados no PyTorch para construir modelos de aprendizado profundo.

# 2. FERRAMENTAS UTILIZADAS 🛠

Neste projeto utilizamos várias bibliotecas Python, cada uma com um propósito específico, algumas delas são:

<ol>
  <li><strong>os</strong>: <code>Interage com o sistema operacional, permitindo a manipulação de arquivos e diretórios.</code></li>
  <li><strong>torch</strong>: <code>PyTorch é usado para aprendizado profundo.</code></li>
  <li><strong>warnings</strong>: <code>Emite mensagens de aviso ao usuário.</code></li>
  <li><strong>math e numpy</strong>: <code>Realizam operações matemáticas.</code></li>
</ol>

## 3. Tensores com Pytorch 🧮

### 3.1 Tensores com Valores Fixos

No PyTorch, `torch.zeros`, `torch.ones` e `torch.full` são funções que retornam tensores preenchidos com valores fixos.

<ol>
  <li>
    <code>torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Este método retorna um tensor preenchido com o valor escalar 0, com a forma definida pelo argumento variável <code>size</code>.
  </li>
  <li>
    <code>torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Este método retorna um tensor preenchido com o valor escalar 1, com a forma definida pelo argumento variável <code>size</code>.
  </li>
  <li>
    <code>torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Este método retorna um tensor de tamanho <code>size</code> preenchido com <code>fill_value</code>.
  </li>
</ol>

Essas funções são úteis para inicializar tensores quando você sabe a forma do tensor, mas não precisa dos valores iniciais. Além disso, elas também são úteis para criar máscaras ou outros tensores auxiliares em seus cálculos.

### 3.2 Transformando Listas e Arrays Numpy em Tensores e Vice-versa

A capacidade de transformar arrays Numpy e listas em tensores e vice-versa é extremamente útil em muitos cenários. Aqui estão alguns exemplos:

<ol>
  <li>
    <strong>Compatibilidade com bibliotecas existentes</strong>: Muitas bibliotecas de ciência de dados e aprendizado de máquina, como NumPy, Pandas e Scikit-learn, usam arrays NumPy como estrutura de dados principal. Ser capaz de converter facilmente entre tensores PyTorch e arrays NumPy permite que você integre código PyTorch com código que usa essas bibliotecas.
  </li>
  <li>
    <strong>Eficiência de memória</strong>: Ao converter arrays NumPy em tensores PyTorch, o PyTorch tentará usar a mesma memória subjacente que o array NumPy, se possível. Isso pode resultar em economia de memória significativa se você estiver trabalhando com grandes arrays NumPy que você deseja converter em tensores.
  </li>
  <li>
    <strong>Aproveitando recursos do PyTorch</strong>: O PyTorch oferece muitos recursos poderosos, como diferenciação automática e aceleração de GPU, que não estão disponíveis no NumPy. Ao converter arrays NumPy em tensores PyTorch, você pode aproveitar esses recursos.
  </li>
  <li>
    <strong>Manipulação de dados</strong>: As listas são uma estrutura de dados fundamental em Python e são usadas para armazenar coleções de itens. Ser capaz de converter facilmente entre listas e tensores permite que você manipule e processe esses dados usando as operações de tensor do PyTorch.
  </li>
</ol>


### 3.3 Tensores aleatórios

Os tensores aleatórios são uma parte importante do PyTorch e são usados em muitos cenários, como a inicialização de pesos em redes neurais. Aqui estão algumas funções que você pode usar para criar tensores aleatórios no PyTorch:

<ol>
  <li>
    <code>torch.rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com números aleatórios uniformemente distribuídos entre 0 e 1.
  </li>
  <li>
    <code>torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com números aleatórios normalmente distribuídos com média 0 e variância 1.
  </li>
  <li>
    <code>torch.randint(low=0, high, size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com inteiros aleatórios gerados uniformemente entre <code>low</code> (inclusive) e <code>high</code> (exclusivo).
  </li>
</ol>

Essas funções são úteis quando você precisa de um tensor com elementos aleatórios. Por exemplo, você pode usar `torch.rand` para inicializar os pesos de uma rede neural com valores aleatórios pequenos, o que é uma prática comum no aprendizado profundo.

### 3.4 Formas de Tensores

A “forma” de um tensor se refere às dimensões do tensor. Por exemplo, um tensor 1D (ou vetor) com comprimento `n` terá a forma `(n,)`. Um tensor 2D (ou matriz) com `m` linhas e `n` colunas terá a forma `(m, n)`. Da mesma forma, um tensor 3D terá a forma `(l, m, n)` e assim por diante.

Aqui estão algumas maneiras de criar tensores com formas específicas no PyTorch:

<ol>
  <li>
    <code>torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com dados não inicializados. A forma do tensor é definida pelo argumento variável <code>size</code>.
  </li>
  <li>
    <code>torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor não inicializado com o mesmo tamanho que o <code>input</code>. É equivalente a <code>torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)</code>.
  </li>
  <li>
    <code>torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com o valor escalar 0, com o mesmo tamanho que o <code>input</code>. É equivalente a <code>torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)</code>.
  </li>
  <li>
    <code>torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com o valor escalar 1, com o mesmo tamanho que o <code>input</code>. É equivalente a <code>torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)</code>.
  </li>
  <li>
    <code>torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False) → Tensor</code>: Esta função retorna um tensor preenchido com números aleatórios de uma distribuição uniforme no intervalo [0, 1), com o mesmo tamanho que o <code>input</code>. É equivalente a <code>torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)</code>.
  </li>
</ol>

### 3.5 Tipos de dados tensores

Os tensores do PyTorch são matrizes multidimensionais que contêm elementos de um único tipo de dado. O PyTorch define 10 tipos de tensores com variantes de CPU e GPU. Aqui estão os tipos de dados de tensor disponíveis no PyTorch 2.0:

<ul>
  <li><strong>32-bit floating point</strong>: <code>torch.float32</code> ou <code>torch.float</code></li>
  <li><strong>64-bit floating point</strong>: <code>torch.float64</code> ou <code>torch.double</code></li>
  <li><strong>16-bit floating point [1]</strong>: <code>torch.float16</code> ou <code>torch.half</code></li>
  <li><strong>16-bit floating point [2]</strong>: <code>torch.bfloat16</code></li>
  <li><strong>32-bit complex</strong>: <code>torch.complex32</code> ou <code>torch.chalf</code></li>
  <li><strong>64-bit complex</strong>: <code>torch.complex64</code> ou <code>torch.cfloat</code></li>
  <li><strong>128-bit complex</strong>: <code>torch.complex128</code> ou <code>torch.cdouble</code></li>
  <li><strong>8-bit integer (unsigned)</strong>: <code>torch.uint8</code></li>
  <li><strong>8-bit integer (signed)</strong>: <code>torch.int8</code></li>
  <li><strong>16-bit integer (signed)</strong>: <code>torch.int16</code> ou <code>torch.short</code></li>
  <li><strong>32-bit integer (signed)</strong>: <code>torch.int32</code> ou <code>torch.int</code></li>
  <li><strong>64-bit integer (signed)</strong>: <code>torch.int64</code> ou <code>torch.long</code></li>
  <li><strong>Boolean</strong>: <code>torch.bool</code></li>
</ul>

Cada tipo de tensor tem suas próprias características e usos. Veja alguns exemplos de criação de tensores abaixo:

## 4. Matemática e lógica com tensores PyTorch 💡

Os tensores do PyTorch suportam uma ampla variedade de operações matemáticas e lógicas. Aqui estão alguns exemplos:

<ul>
  <li><strong>Operações aritméticas</strong>: As operações aritméticas básicas como adição, subtração, multiplicação e divisão podem ser realizadas em tensores. Além disso, o PyTorch também suporta operações mais complexas como raiz quadrada, exponenciação, logaritmo, etc.</li>
  <li><strong>Operações lógicas</strong>: O PyTorch suporta operações lógicas como AND, OR e XOR em tensores booleanos. Por exemplo, a função <code>torch.logical_and</code> calcula o AND lógico elemento a elemento dos tensores de entrada. Os zeros são tratados como False e os não zeros são tratados como True.</li>
  <li><strong>Indexação, fatiamento, junção, mutação</strong>: O PyTorch suporta várias operações para manipular tensores, incluindo indexação, fatiamento, junção (por exemplo, <code>torch.cat</code>), e mutação (por exemplo, <code>torch.transpose</code>). Essas operações permitem reorganizar, redimensionar, e modificar tensores de várias maneiras.</li>
  <li><strong>Funções matemáticas</strong>: O PyTorch também inclui uma ampla gama de funções matemáticas que podem ser aplicadas a tensores, como funções trigonométricas, funções exponenciais e logarítmicas, etc.</li>
</ul>

Essas operações podem ser usadas para realizar uma ampla variedade de cálculos e são fundamentais para muitos algoritmos de aprendizado de máquina.

### 4.1 Operações Aritméticas

As operações aritméticas são fundamentais quando trabalhamos com tensores no PyTorch. Aqui estão algumas das operações mais comuns:

<ol>
  <li>
    <strong>Adição</strong>: Você pode adicionar dois tensores usando o operador <code>+</code> ou a função <code>torch.add()</code>. Por exemplo, se você tem dois tensores <code>a</code> e <code>b</code>, você pode adicionar os dois usando <code>a + b</code> ou <code>torch.add(a, b)</code>.
  </li>
  <li>
    <strong>Subtração</strong>: A subtração de tensores pode ser realizada usando o operador <code>-</code> ou a função <code>torch.sub()</code>. Por exemplo, <code>a - b</code> ou <code>torch.sub(a, b)</code>.
  </li>
  <li>
    <strong>Multiplicação</strong>: A multiplicação de tensores pode ser feita de várias maneiras, dependendo do que você precisa. A multiplicação elemento a elemento pode ser feita usando <code>a * b</code> ou <code>torch.mul(a, b)</code>. A multiplicação de matrizes pode ser realizada usando <code>torch.matmul(a, b)</code>.
  </li>
  <li>
    <strong>Divisão</strong>: A divisão de tensores pode ser realizada usando o operador <code>/</code> ou a função <code>torch.div()</code>. Por exemplo, <code>a / b</code> ou <code>torch.div(a, b)</code>.
  </li>
</ol>

### 4.2 Manipulando formas de tensor

Trabalhar com tensores em PyTorch muitas vezes envolve manipular suas formas. Aqui estão algumas das operações mais comuns:

<ol>
  <li>
    <strong>Reshape</strong>: A função <code>torch.reshape()</code> pode ser usada para reorganizar os elementos de um tensor para se ajustar a uma determinada forma. Por exemplo, se você tem um tensor de forma <code>(4, 2)</code> e deseja reorganizá-lo para a forma <code>(2, 4)</code>, você pode usar <code>torch.reshape(tensor, (2, 4))</code>.
  </li>
  <li>
    <strong>Squeeze e Unsqueeze</strong>: <code>torch.squeeze()</code> remove as dimensões de tamanho 1 do tensor, enquanto <code>torch.unsqueeze()</code> adiciona uma dimensão extra de tamanho 1. Isso é útil para adicionar ou remover dimensões que são necessárias para certas operações.
  </li>
  <li>
    <strong>Flatten</strong>: <code>torch.flatten()</code> é usado para transformar o tensor em um tensor 1D. Isso é útil quando você quer transformar um tensor multidimensional em um vetor.
  </li>
  <li>
    <strong>Permute e Transpose</strong>: <code>torch.permute()</code> permite reordenar as dimensões de um tensor de qualquer maneira que você quiser. <code>torch.transpose()</code> é um caso especial disso, onde duas dimensões são trocadas. Isso é comumente usado para trocar as dimensões de altura e largura em imagens.
  </li>
  <li>
    <strong>Size e Shape</strong>: <code>tensor.size()</code> e <code>tensor.shape</code> retornam o tamanho do tensor. <code>tensor.numel()</code> retorna o número total de elementos no tensor.
  </li>
</ol>

### 4.3 Fatiamento/Slicing de Tensores

O fatiamento, ou slicing, é uma técnica essencial ao trabalhar com tensores no PyTorch. Esta operação permite acessar e manipular partes específicas de um tensor de forma eficiente e intuitiva.

Semelhante às listas e arrays em Python, o PyTorch permite o fatiamento de tensores. Por exemplo, em um tensor bidimensional (uma matriz), é possível acessar um elemento específico fornecendo os índices da linha e da coluna. Além disso, é possível acessar uma linha ou coluna inteira de uma matriz usando a técnica de fatiamento.

O fatiamento é uma ferramenta poderosa que facilita a manipulação e o acesso aos dados dos tensores. Com a prática, você descobrirá muitos usos para o fatiamento ao trabalhar com tensores no PyTorch.



