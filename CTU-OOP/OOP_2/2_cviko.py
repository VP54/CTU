class CreateUsernames:
    """ Class that solves second OOP-API assignment """

    def __init__(self):
        self.data = data

    def __filter_active(self):

        student_lst = self.data.get('students')
        active_lst  = self.data.get('active')
        
        """ Iterates in parallel over lists and removes names that are non active """

        return [
            name for is_active, name                                                                    \
            in zip(active_lst, student_lst)                                                             \
            if is_active]

    def __parse_names(self):
        
        student_lst = self.__filter_active()
        parsed_names = []

        for student_name in student_lst:
            lst = student_name.split()
            first_name = lst[0][0:3].lower()
            last_name = lst[1][0:5].lower()
            
            parsed_names.append(last_name + first_name)
        
        return  sorted(parsed_names),                                                                   \
                [name for _, name in sorted(zip(parsed_names, student_lst))]                                                                          

    def __enum_unique(self):

        """ 
        Enumerates parsed data (that and only that is an unique key)
            1. In sorted lists looks at previous value 
                a. if equal searches for number
                b. else append '2'
            2. returns result
        """

        parsed_names, student_names = self.__parse_names()

        for index, name in enumerate(parsed_names):
            if (index > 0)                                                                              \
                and                                                                                     \
                (parsed_names[index][0:8] == parsed_names[index - 1][0:8]):

                digit = ''

                for letter in parsed_names[index - 1]:
                    if letter.isdigit():
                        digit = digit + letter
                        
                if digit:
                    digit = int(digit) + 1
                    parsed_names[index] += str(digit)
                else:
                    parsed_names[index] += "2"
        
        return parsed_names, student_names

    def return_dict(self):
        """ Ensembles all lists and created dictionary """

        parsed_names, student_names = self.__enum_unique()

        return {
                'students': student_names,
                'active': [True for _ in enumerate(parsed_names)],
                'usernames': parsed_names
        
            }


# Tests

vp = ["Vaclav Pazderka" for _ in range(100)]
jd = ["John Deere" for _ in range(100)]
al = ["Adam Levine" for _ in range(100)]
mm = ["Monica Muller" for _ in range(100)]

v = [True for _ in range(100)]
j = [True for _ in range(100)]
a = [True for _ in range(100)]
m = [False for _ in range(100)]

data = {
    "students": vp + mm + jd + al,
    "active": v + m + j + a
}

create_user = CreateUsernames()

result = create_user.return_dict()







#TESTING 


import pandas as pd

pd.set_option("display.max_rows", 1500)

df = pd.DataFrame(result)

print(df)