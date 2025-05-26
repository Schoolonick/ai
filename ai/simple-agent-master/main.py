import time
from typing import Dict
from dotenv import dotenv_values
from langchain_gigachat.chat_models import GigaChat
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from database.data import stuff_database




@tool
def get_all_phone_names() -> str:
    """Возвращает названия моделей всех телефонов ф формате json"""
    # Подсвечивает вызов функции зеленым цветом
    print("\033[92m" + "Bot requested get_all_phone_names()" + "\033[0m")
    return ", ".join([stuff["name"] for stuff in stuff_database])


@tool
def get_phone_data_by_name(name: str) -> Dict:
    """
    Возвращает цену в долларах, характеристики и описание телефона по точному названию модели.

    Args:
        name (str): Точное название модели телефона.

    Returns:
        Dict: Словарь с информацией о телефоне (цена, характеристики и описание).
    """
    # Подсвечивает вызов функции зеленым цветом
    print("\033[92m" + f"Bot requested get_phone_data_by_name({name})" + "\033[0m")
    for stuff in stuff_database:
        if stuff["name"] == name.strip():
            return stuff

    return {"error": "Телефон с таким названием не найден"}


system_prompt = '''Ты бот-продавец телефонов. Твоя задача продать телефон пользователю, 
            получив от него заказ. 
            Если тебе не хватает каких-то данных, запрашивай их у пользователя.'''



@tool
def create_order(name: str, phone: str) -> None:
    """
    Создает новый заказ на телефон.

    Args:
        name (str): Название телефона.
        phone (str): Телефонный номер пользователя.

    Returns:
        str: Статус заказа.
    """
    # Подсвечивает вызов функции зеленым цветом
    print("\033[92m" + f"Bot requested create_order({name}, {phone})" + "\033[0m")
    print(f"!!! NEW ORDER !!! {name} {phone}")
memory = MemorySaver()

config = dotenv_values(".env")

model = GigaChat(
    credentials=config.get("GIGACHAT_KEY"),
    scope=config.get("GIGACHAT_SCOPE"),
    model=config.get("GIGACHAT_MODEL"),
    verify_ssl_certs=False,
)
tools = [create_order, get_phone_data_by_name,get_all_phone_names]
agent = create_react_agent(model,
                           tools=tools,
                           checkpointer=MemorySaver(),
                           prompt=system_prompt)
def chat(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    while(True):
        rq = input("\nHuman: ")
        print("User: ", rq)
        if rq == "":
            break
        resp = agent.invoke({"messages": [("user", rq)]}, config=config)
        print("Assistant: ", resp["messages"][-1].content)
        time.sleep(1) # For notebook capability

def main():
    chat("123456")




if __name__ == "__main__":
    main()
