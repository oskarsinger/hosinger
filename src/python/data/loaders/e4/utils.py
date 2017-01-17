def get_symptom_status(subject):

    status = None

    if type(subject) in {str, unicode}:
        if len(subject) > 2:
            subject = subject[-2:]
        
        subject = int(subject)

    # Symptomatic
    Hsx = {2, 4, 5, 8, 9, 11, 17, 18, 19, 20, 23}

    # Asymptomatic
    Lsx = {6, 7, 12, 13, 21, 22, 24}

    # Wild type
    W = {3}

    if subject in Hsx:
        status = 'Hsx'
    elif subject in Lsx:
        status = 'Lsx'
    elif subject in W:
        status = 'W'
    else:
	status = 'U'

    return status

