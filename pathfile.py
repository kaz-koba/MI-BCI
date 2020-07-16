class PATHfile:
    def __init__(self, name, day, trial):
        self.name = name
        self.day = day
        self.trial = trial

    def edfpath(name, day, trial):
        return 'EDFfile/record-' + day + '-' + trial + '-' + name + '.edf'
    
    def eventpath(name, day, trial):
        return 'eventdata/event-record-' + day + '-' + trial + '-' + name + '.csv'
        