from cryptography.fernet import Fernet
import os

def generate_key():
    key_path = 'secret.key'
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
        print(f"New key created and saved to {key_path}")
    else:
        print(f'Key already exists: {key_path}')

def load_key():
    key_path = 'secret.key'
    if not os.path.exists(key_path):
        setup_encryption()
    with open(key_path, 'rb') as key_file:
        return key_file.read()

def encrypt_data(api_key, account_id):
    generate_key()
    key = load_key()
    fernet = Fernet(key)
    
    encrypted_api_key = fernet.encrypt(api_key.encode())
    encrypted_account_id = fernet.encrypt(account_id.encode())
    
    with open('encrypted_data.txt', 'wb') as encrypted_file:
        encrypted_file.write(encrypted_api_key + b'\n' + encrypted_account_id)
    print("Data encrypted and saved to encrypted_data.txt")

def decrypt_data():
    key = load_key()
    fernet = Fernet(key)

    with open('encrypted_data.txt', 'rb') as encrypted_file:
        encrypted_lines = encrypted_file.readlines()
        api_key = fernet.decrypt(encrypted_lines[0].strip()).decode()
        account_id = fernet.decrypt(encrypted_lines[1].strip()).decode()

    return api_key, account_id

def setup_encryption():
    api_key = input("Enter your OANDA API key: ")
    account_id = input("Enter your OANDA account ID: ")
    encrypt_data(api_key, account_id)