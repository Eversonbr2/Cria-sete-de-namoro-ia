# Cria-sete-de-namoro-ia
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carregar modelo de linguagem (você pode trocar por outros modelos como 'mistralai/Mistral-7B-Instruct-v0.1')
model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None
step = 0

print("👩❤️ Olá, amor! Sou sua namorada virtual. Como foi o seu dia? (Digite 'sair' para encerrar)")

while True:
    user_input = input("Você: ")
    if user_input.lower() in ['sair', 'exit', 'quit']:
        print("👩 Até logo, meu amor. Vou sentir sua falta! ❤️")
        break

    # Codificar entrada do usuário
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Adicionar ao histórico de conversa
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if step > 0 else new_input_ids

    # Gerar resposta
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decodificar e exibir a resposta
    resposta = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Namada ❤️:", resposta)

    step += 1
    
