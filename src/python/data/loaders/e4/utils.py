def get_symptom_status(subject):

    status = None

    if type(subject) in {str, unicode}:
        if len(subject) > 2:
            subject = subject[-2:]
        
        subject = int(subject)

    # Symptomatic
    Sx = {2, 4, 5, 8, 9, 11, 17, 18, 19, 20, 23}

    # Asymptomatic
    Asx = {6, 7, 12, 13, 21, 22, 24}

    # Wild type
    W = {3}

    if subject in Sx:
        status = 'Sx'
    elif subject in Asx:
        status = 'Asx'
    elif subject in W:
        status = 'W'
    else:
	status = 'U'

    return status

