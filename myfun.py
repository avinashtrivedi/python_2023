import random
import string
def get_full_name(first_name, last_name, password):
    full_name = f"{first_name} {last_name} {password}"
    return full_name

def generate_random_email(length):
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    email = f"test{random_part}@example.com"
    return email
