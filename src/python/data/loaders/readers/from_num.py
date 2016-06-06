def get_magnitude(num_list):

    squares = [axis**2 for axis in num_list]

    return (sum(squares))**(0.5)

def get_scalar_as_is(num):

    return num

def get_vec_as_list(nums):

    return [num for num in nums]
