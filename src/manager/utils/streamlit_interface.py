from src.manager.utils.singleton import singleton

def get_user_message(message):
    user_input = input(message)
    return user_input

def output_assistant_response(response):
    print(response)