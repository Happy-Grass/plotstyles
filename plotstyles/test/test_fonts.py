import json


json_path = './plotstyles/styles/fonts/setting.json'

with open(json_path, 'r') as json_file:
    font_data = json.load(json_file)
print(font_data)

family_serif = font_data.get('serif')
family_sansserif = font_data.get('sans-serif')
properties = font_data.get('fontproperties')
print(properties)