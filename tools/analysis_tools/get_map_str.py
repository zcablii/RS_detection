"""
python tools/analysis_tools/get_map_str.py
"""

test_str = "Fri Sep 23 10:55:07 2022  Validation Score:0.810597, Airplane_AP:0.9855661, Ship_AP:0.8121692, Vehicle_AP:0.9415818, Basketball_Court_AP:0.8330959, Tennis_Court_AP:0.9512171, Football_Field_AP:0.8432544, Baseball_Field_AP:0.9506423, Intersection_AP:0.7516665, Roundabout_AP:0.9369346, Bridge_AP:0.0998421, meanAP:0.8105970, iter:41609"



Vehicle = test_str.find("Vehicle_AP:") + len("Vehicle_AP:")
Airplane = test_str.find("Airplane_AP:") + len("Airplane_AP:")
Ship = test_str.find("Ship_AP:") + len("Ship_AP:")
Intersection = test_str.find("Intersection_AP:") + len("Intersection_AP:")
Tennis = test_str.find("Tennis_Court_AP:") + len("Tennis_Court_AP:")
Basketball = test_str.find("Basketball_Court_AP:") + len("Basketball_Court_AP:")
Bridge = test_str.find("Bridge_AP:") + len("Bridge_AP:")
Baseball = test_str.find("Baseball_Field_AP:") + len("Baseball_Field_AP:")
Football = test_str.find("Football_Field_AP:") + len("Football_Field_AP:")
Roundabout = test_str.find("Roundabout_AP:") + len("Roundabout_AP:")
meanAP = test_str.find("meanAP:") + len("meanAP:")


print("Number Order:\n")

print(test_str[Vehicle:Vehicle+6], " - Vehicle")
print(test_str[Airplane:Airplane+6], " - Airplane")
print(test_str[Ship:Ship+6], " - Ship")
print(test_str[Intersection:Intersection+6], " - Intersection")
print(test_str[Tennis:Tennis+6], " - Tennis")
print(test_str[Basketball:Basketball+6], " - Basketball")
print(test_str[Bridge:Bridge+6], " - Bridge")
print(test_str[Baseball:Baseball+6], " - Baseball")
print(test_str[Football:Football+6], " - Football")
print(test_str[Roundabout:Roundabout+6], " - Roundabout")
print(test_str[meanAP:meanAP+6], " - meanAP")


print("\nCLASSES Order:\n")
print(test_str[Airplane:Airplane+6], " - Airplane")
print(test_str[Ship:Ship+6], " - Ship")
print(test_str[Vehicle:Vehicle+6], " - Vehicle")
print(test_str[Basketball:Basketball+6], " - Basketball")
print(test_str[Tennis:Tennis+6], " - Tennis")
print(test_str[Football:Football+6], " - Football")
print(test_str[Baseball:Baseball+6], " - Baseball")
print(test_str[Intersection:Intersection+6], " - Intersection")
print(test_str[Roundabout:Roundabout+6], " - Roundabout")
print(test_str[Bridge:Bridge+6], " - Bridge")
print(test_str[meanAP:meanAP+6], " - meanAP")


# 0.9895  - Airplane
# 0.8376  - Ship
# 0.9457  - Vehicle
# 0.9017  - Basketball
# 0.9612  - Tennis
# 0.8877  - Football
# 0.9413  - Baseball
# 0.7948  - Intersection
# 0.9152  - Roundabout
# 0.5289  - Bridge
# 0.8704  - meanAP

#  399336.188  2009072.500  288124.031  234.621  2323.610  772.123
#   7510.074  33430.688  19.482  354444.188  6422710.500

# | 类别             | 数量   | CE | EQLv2 | Seesaw | Group Softmax | BCE | EFL | Re-EFL | Soft-IoU |
# |------------------|--------|----|-------|--------|---------------|-----|-----|--------|----------|
# | Airplane         | 10,671 |    |       |        |               |     |     |        |          |
# | Ship             | 8,689  |    |       |        |               |     |     |        |          |
# | Vehicle          | 66,017 |    |       |        |               |     |     |        |          |
# | Basketball_Court | 394    |    |       |        |               |     |     |        |          |
# | Tennis_Court     | 731    |    |       |        |               |     |     |        |          |
# | Football_Field   | 236    |    |       |        |               |     |     |        |          |
# | Baseball_Field   | 252    |    |       |        |               |     |     |        |          |
# | Intersection     | 1,549  |    |       |        |               |     |     |        |          |
# | Roundabout       | 136    |    |       |        |               |     |     |        |          |
# | Bridge           | 311    |    |       |        |               |     |     |        |          |




# 0.9913
# 0.8317
# 0.9411
# 0.8960
# 0.9352
# 0.8653
# 0.9410
# 0.7937
# 0.9098
# 0.5651
# 0.8670






