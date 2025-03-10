# Sudoku-PTBR
Este é um script Python para resolver um Sudoku a partir de um arquivo de entrada.

## Como Rodar o Script

### Pré-requisitos:
  - Python 3.x instalado em seu computador.
  - Arquivo de entrada contendo o Sudoku a ser resolvido (no formato de texto, como `entrada.txt`).

### Passos para rodar o script:
  1. Abra o terminal ou prompt de comando no diretório onde o arquivo `sudoku-ptbr.py` está localizado.
  
  2. Utilize o seguinte comando para rodar o script:
  
     ```bash
     py .\\sudoku-ptbr.py entrada.txt
     ```

   Onde:
   - `.\sudoku-ptbr.py` é o script que irá resolver o Sudoku.
   - `entrada.txt` é o arquivo de entrada que contém o Sudoku a ser resolvido.

### Exemplo de Arquivo de Entrada:
   O arquivo `entrada.txt` deve conter os números do Sudoku dispostos em uma única linha, separados por espaços. Por exemplo:

  5 3 0 0 7 0 0 0 0 6 0 0 1 9 5 0 0 0 0 9 8 0 0 0 0 6 0 8 0 0 0 6 0 0 0 3 4 0 0 8 0 3 0 0 1 7 0 0 0 2 0 0 0 6 0

### Saída:
O script irá imprimir a solução do Sudoku no terminal e gerar um arquivo `solucao.txt`.
