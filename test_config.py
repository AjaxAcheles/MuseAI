from core.config_loader import load_config
c = load_config('config.yaml')
print('Config loaded successfully')
print('Planner model:', c.endpoints.planner.model_name)
print('Base URL:', c.endpoints.planner.base_url)