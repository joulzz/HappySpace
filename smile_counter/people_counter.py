class PeopleTracker:
    def __init__(self):
        self.total_detected_bboxes = []
        self.previous_frame_bboxes = []
        self.current_frame_bboxes = []
        self.replaced_indices = []

class PeopleCounter:
    """
    Counter class to hold people instances
    """
    def __init__(self):
        self.people = []

    def add(self, people_instance):
        self.people.append(people_instance)

    def remove(self, people_instance):
        for people in self.people:
            if people.id == people_instance.id:
                people.bbox = []

class People:
    """"
    Class to represent people attributes
    """
    def __init__(self):
        self.id = 0
        self.current = False
        self.bbox = []
        self.count = None
        self.history = []
        self.timestamp = None
        self.gps = None
        self.non_smiles = 0
        self.gender = None
        self.age = 0
