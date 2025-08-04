# Agente Neural para Survival Game

Este projeto implementa um agente baseado em rede neural simples que pode ser otimizado usando Grey Wolf Optimizer (GWO).

## Arquivos Criados

1. **`neural_agent.py`** - Implementação dos agentes neurais
2. **`gwo_neural_trainer.py`** - Treinador GWO para agentes neurais  
3. **`compare_agents.py`** - Script para comparar diferentes tipos de agentes
4. **`gwo.py`** - GWO atualizado com suporte a limites personalizados

## Como Usar

### 1. Treinar o Agente Neural

```bash
python gwo_neural_trainer.py
```

Isso irá:
- Criar uma rede neural simples (entrada → 6 neurônios ocultos → 3 saídas)
- Usar GWO para otimizar os pesos da rede
- Salvar os melhores pesos em `best_neural_weights.npy`
- Testar o agente treinado

### 2. Testar o Agente Treinado

```bash
python gwo_neural_trainer.py test
```

### 3. Demonstração Visual

```bash
python gwo_neural_trainer.py demo
```

### 4. Comparar Todos os Agentes

```bash
python compare_agents.py
```

Compara:
- RuleBasedAgent padrão
- RuleBasedAgent otimizado (se disponível)
- Neural Agent otimizado (se disponível)

### 5. Demonstração do Melhor Agente

```bash
python compare_agents.py demo
```

## Diferenças dos Agentes

### RuleBasedAgent
- **Estratégia fixa**: Sempre segue as mesmas regras
- **Parâmetros otimizáveis**: Apenas 3 valores que ajustam quando as regras são aplicadas
- **Limitação**: Nunca pode aprender estratégias além das programadas

### Neural Agent
- **Estratégia flexível**: Pode aprender qualquer mapeamento estado→ação
- **Parâmetros otimizáveis**: Todos os pesos da rede neural (39 pesos na versão simples)
- **Potencial**: Pode descobrir estratégias completamente novas

## Arquitetura da Rede Neural

### SimpleNeuralAgent
- **Entrada**: Grid de sensores (25 valores) + posição Y (1 valor) = 26 entradas
- **Camada oculta**: 6 neurônios com ativação tanh
- **Saída**: 3 neurônios (uma para cada ação: parar, subir, descer)
- **Total de pesos**: 26×6 + 6 + 6×3 + 3 = 189 parâmetros

### Função de Ativação
- **Camada oculta**: tanh (permite valores negativos)
- **Saída**: Linear com argmax (escolhe a ação com maior valor)

## Configurações do GWO

- **População**: 30 lobos
- **Iterações**: 200
- **Limites dos pesos**: [-3, 3]
- **Testes por avaliação**: 3 jogos por lobo

## Por que o Neural Agent é Melhor?

1. **Flexibilidade**: Pode aprender padrões complexos nos dados
2. **Adaptabilidade**: Não limitado por regras pré-definidas
3. **Capacidade de generalização**: Pode reagir a situações não previstas
4. **Otimização global**: 189 parâmetros vs 3 do RuleBasedAgent

## Exemplo de Uso

```python
from neural_agent import SimpleNeuralAgent
from game.core import GameConfig
import numpy as np

# Carregar agente treinado
config = GameConfig()
weights = np.load("best_neural_weights.npy")
agent = SimpleNeuralAgent(config, weights)

# Usar no jogo
state = game.get_state(0, include_internals=True)
action = agent.predict(state)
```

## Arquivos de Saída

- `best_neural_weights.npy` - Melhores pesos da rede neural
- `best_weights.npy` - Melhores parâmetros do RuleBasedAgent (se existir)

## Próximos Passos

1. Experimentar com diferentes arquiteturas de rede
2. Implementar redes neurais mais profundas
3. Usar algoritmos evolutivos mais sofisticados
4. Adicionar técnicas de regularização
