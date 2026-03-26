# class People:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# people = [People("Bob", 18), People("Alice", 19)]


people = [
    {"name": "Bob", "age": 18},
    {"name": "Jack", "age": 39},
    {"name": "Lily", "age": 32},
    {"name": "Alice", "age": 19},
]


# for person in people:
#     logger.info(f"Name: {person['name']}, Age: {person['age']}")


# new_people = sorted(people, key=lambda person: person["age"])

# logger.info(new_people)

# for index, person in enumerate(people):
#     logger.info(f"Index: {index}, Name: {person['name']}, Age: {person['age']}")

arr = [2, 3, 5, 4, 2, 1]

# for element in arr:
#     logger.info(element)

# for index, element in enumerate(arr):
#     logger.info(f"Index: {index}, Element: {element}")

user = {"name": "Bob", "age": 18, "address": "Shanghai"}

# for key, value in user.items():
#     logger.info(f"Key: {key}, Value: {value}")

for key in user:
    logger.info(f"Key: {key}")

for value in user.values():
    logger.info(f"Value: {value}")

for key, value in user.items():
    logger.info(f"Key: {key}, Value: {value}")
