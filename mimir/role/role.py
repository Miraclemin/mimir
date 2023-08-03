import json
from typing import Dict, List, Optional, Tuple

class Role:
    def __init__(self, json_file):
        self.json_file = json_file
        self.all_roles_name = []
        self.all_roles_prompts = []
        self.all_roles = {}
        self.get_all_roles()

    def get_all_roles(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        roles = data.get('roles', [])
        for role in roles:
            role_name = role.get('role_name')
            self.all_roles_name.append(role_name)
            role_prompt = role.get('role_prompt')
            self.all_roles_prompts.append(role_prompt)
            self.all_roles[role_name] = role_prompt

