class PATHfile:
    def __init__(self, name, day, trial):
        self.name = name
        self.day = day
        self.trial = trial

    def edfpath(name, day, trial):
        return 'C:/Users/sprin/Desktop/test/EDFfile/record-' + day + '-' + trial + '-' + name + '.edf'
    
    def eventpath(name, day, trial):
        return 'C:/Users/sprin/Desktop/test/eventdata/event-record-' + day + '-' + trial + '-' + name + '.csv'

    def listpath(name, day, trial):
        return 'C:/Users/sprin/Desktop/test/listdata/event-record-' + day + '-' + trial + '-' + name + '.csv'
        