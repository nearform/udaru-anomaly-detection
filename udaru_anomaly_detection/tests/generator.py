
from faker import Faker

seeds = {
    'train': 5745349561985196,
    'test': 7256578370237522
}


def generate_resource(size: int, dataset: str):
    fake = Faker()
    fake.seed_instance(seeds[dataset])

    for i in range(size):
        resource = f'res:{fake.sha1()[:8]}'
        if (fake.boolean(chance_of_getting_true=50)):
            resource += f':{fake.first_name().lower()}'

        if (fake.boolean(chance_of_getting_true=90)):
            resource += f':/{fake.language_code()}'

        if (fake.boolean(chance_of_getting_true=40)):
            resource += f'/{fake.user_name()}'

        if (fake.boolean(chance_of_getting_true=20)):
            resource += f'/{fake.domain_word()}'

        yield resource
