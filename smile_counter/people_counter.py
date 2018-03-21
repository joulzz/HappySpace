class PeopleTracker:
    def __init__(self):
        self.total_detected_bboxes = []
        self.previous_frame_bboxes = []
        self.current_frame_bboxes = []

    def add_to_total(self, person):
        self.total_detected_bboxes.append(person)
        self.people_detected += 1

    def add_to_previous(self, person):
        self.previous_frame_bboxes.append(person)

    def remove_from_previous(self, person):
        self.remove_from_previous(person)

    def remove_from_total(self, person):
        self.total_detected_bboxes.remove(person)
        self.people_detected -= 1

    def replace_previous_frame(self, old_person, new_person):
        self.replaced_indices = []
        for i, person in enumerate(self.previous_frame_bboxes):
            if old_person == person:
                self.previous_frame_bboxes[i] = new_person

    def replace_total_detected(self, old_person, new_person):
        for i, person in enumerate(self.total_detected_bboxes):
            if old_person == person:
                self.previous_frame_bboxes[i] = new_person

    def update(self):
        self.previous_frame_bboxes = self.current_frame_bboxes
#
class PeopleCounter:
    def __init__(self):
        self.people_features = []

    def add(self, people_features):
        self.people_features.append(people_features)

    def remove(self, people_features):
        for people in self.people_features:
            if people.id == people_features.id:
                people.bbox = []

class PeopleFeatures:
    def __init__(self):
        self.id = 0
        self.current = False
        self.bbox = []
        self.count = 0
