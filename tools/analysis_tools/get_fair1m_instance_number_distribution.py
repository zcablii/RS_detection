import os
import os.path as osp
import xml.etree.ElementTree as ET
import pprint
pp = pprint.PrettyPrinter(depth=4)

data_root = "~/Downloads/FAIR1M2.0"
sub_dirs = ['train', 'validation']

distributions = dict()
total_imgs = 0
for sub_dir in sub_dirs:
    sub_dir_path = osp.expanduser(osp.join(data_root, sub_dir))
    xml_files = [f for f in os.listdir(sub_dir_path)
                 if osp.isfile(osp.join(sub_dir_path, f))]
    for xml_file in xml_files:
        if not xml_file.endswith('.xml'):
            continue
        total_imgs += 1
        xml_path = osp.join(sub_dir_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        if len(root.findall('objects')) != 1:
            raise ValueError('xml file {} has more than one objects'.format(xml_file))
        objects = root.find('objects')
        for obj in objects.findall('object'):
            possible_result = obj.find('possibleresult')
            cls_name = possible_result.find('name').text
            if cls_name not in distributions:
                distributions[cls_name] = 1
            else:
                distributions[cls_name] += 1

pp.pprint(distributions)
total_instances = sum(distributions.values())
print('\ntotal_imgs:', total_imgs)
print('\ntotal_instances:', total_instances)

jdet_coarse2fine_mappings = {
    "Airplane": [
        "A220", "A321", "A330", "A350", "ARJ21", "Boeing737", "Boeing747",
        "Boeing777", "Boeing787", "C919", "other-airplane"],
    "Ship": [
        "Tugboat", "other-ship", "Liquid Cargo Ship", "Motorboat",
        "Passenger Ship", "Dry Cargo Ship", "Warship", "Engineering Ship",
        "Fishing Boat"],
    "Vehicle": [
        "other-vehicle", "Bus", "Cargo Truck", "Small Car", "Dump Truck",
        "Van", "Excavator", "Tractor", "Trailer", "Truck Tractor"],
    "Basketball Court": ["Basketball Court"],
    "Tennis Court": ["Tennis Court"],
    "Football Field": ["Football Field"],
    "Baseball Field": ["Baseball Field"],
    "Intersection": ["Intersection"],
    "Roundabout": ["Roundabout"],
    "Bridge": ["Bridge"],
}

fair1m2jdet_distributions = {}
for key, value in jdet_coarse2fine_mappings.items():
    if key not in fair1m2jdet_distributions:
        fair1m2jdet_distributions[key] = 0
    for v in value:
        if v in distributions:
            fair1m2jdet_distributions[key] += distributions[v]

pp.pprint(fair1m2jdet_distributions)
total_fair1m2jdet_instances = sum(fair1m2jdet_distributions.values())
print('\ntotal_fair1m2jdet_instances:', total_fair1m2jdet_instances)

for plane in jdet_coarse2fine_mappings["Airplane"]:
    print(plane, distributions[plane])

print('\n')

for ship in jdet_coarse2fine_mappings["Ship"]:
    print(ship, distributions[ship])

print('\n')

for vehicle in jdet_coarse2fine_mappings["Vehicle"]:
    print(vehicle, distributions[vehicle])