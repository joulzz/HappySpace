class PeopleTracker:
    def __init__(self):
        self.total_detected_bboxes = []
        self.previous_frame_bboxes = []
        self.current_frame_bboxes = []
        self.replaced_indices = []

class PeopleCounter:
    def __init__(self):
        self.people = []

    def add(self, people_features):
        self.people.append(people_features)

    def remove(self, people_features):
        for people in self.people:
            if people.id == people_features.id:
                people.bbox = []

class People:
    def __init__(self):
        self.id = 0
        self.current = False
        self.bbox = []
        self.count = 0
