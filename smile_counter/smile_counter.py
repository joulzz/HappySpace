class SmileCounter:
    def __init__(self):
        self.smiles_array = []

    def set_people(self, people):
        self.smiles_array = []
        [self.smiles_array.append(0) for bbox in people]

    def add(self):
        self.smiles_array.append(0)

    def add_smile(self, face, people, i):
        if ( face[0][0] + face[1][0])/2 > people[0][0] and ( face[0][0] + face[1][0])/2 < people[1][0] and \
                                    ( face[0][1] + face[1][1])/2 > people[1][1] and ( face[0][1] + face[1][1])/2 < people[1][0]:
            self.smiles_array[i] += 1


