# 1. Write an if statement that checks if a variable is equal to 10. If it is, print "The variable is 10."

x = 10

if x == 10:
    print("The variable is 10.")

# 2. Expand on the previous code. If the variable is not 10, use an else statement to print "The variable is not 10."

x = 9

if x == 10:
    print("The variable is 10.")
else:
    print("The variable is not 10.")

# 3. Now, let's add an elif statement in between. If the variable is greater than 10, print "The variable is greater than 10."

x = 11

if x == 10:
    print("The variable is 10.")
elif x > 10:
    print("The variable is greater than 10.")
else:
    print("The variable is not 10.")

# 4. Write an if statement that checks if a string is equal to "Python". If it is, print "I am learning Python."

string = "Python"

if string == "Python":
    print("I am learning Python.")

# 5. Write a nested if statement (an if statement inside another if statement). First check if a variable is greater than 10. If this is true, check if the same variable is less than 20. If this is also true, print "The variable is between 10 and 20."

x = 15

if x > 10:
    if x < 20:
        print("The variable is between 10 and 20.")
