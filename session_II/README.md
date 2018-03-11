Date: 8.3.2018

Topics
------


 * File handling: opening files with 'with'
 * 'git' by example: git init', 'git add', 'git commit', 'git push origin master'
 * 'if __name__ == "__main__"'
 * 'return' vs 'yield'
 * 'try-except-finally'


### File handling
There are two methods for 'correctly' handling file opening in python. The first uses the try-except-finally error handling introduced above.

    try:
        fid=open(file_name,'r')
    except OSError as e:
        print(e)
    finally:
        fid.close()

The finally statement here means that no matter what the error is the file is always closed correctly.

Alternatively, you can use a 'with' statement.

    with open(file_name,'r') as fid:
        # do stuff

By using a 'with' statement you tell python that if there is an error when opening the file, close the file correctly BEFORE you die. 

Neither method is better or worse than the other, although you may find one method works better in certain circumstances. As an example of reading all the lines from a file into a list and splitting the line into whitespace delimited sections.:

    with open(file_name,'r') as fid:
        lines=[line.split() for line in fid.readlines()]

    for line in lines:
        """Do stuff with your line"""





### 'git' by example

Let's work together on this project! And let's use `git` for that! Here comes a step-by-step explanation how you can help to improve 'byoc' and best part is: You will become a 'git' expert at the same time!

#### Step 1:
Create an account on github.com

#### Step 2:
Create a fork of [byoc](https://github.com/HerrMuellerluedenscheid/byoc) (click on "fork" in the upper right corner)

#### Step 3:
If you haven't cloned this repository yet, now is the time to do so. Checkout the top level page on how to do that [byoc](https://github.com/HerrMuellerluedenscheid/byoc).

Now, you have to add the __remote__ link to the repository on your computer.To do that run something similar to this commant where XXXXXXXX is your account name:

    git remote add mygithub git@github.com:XXXXXXXX/byoc.git

(note that 'origin' will still point to my 'byoc' github repository)

Check your remote connections running:

    git remote -v

#### Step 4a:

Now you are setup to work with git. Before you start working you should always pull the latest changes from the origin repository and possibly from your own if you are working on different computers. Just run the following:

    git pull origin master

and possibly

    git pull mygithub master

to pull any updates.

#### Step 4:

Improve any of the files here, or create something...

#### Step 5:

After you finished working and you think you are good to go, no matter if you changed 1 or 100 lines of code, run:

    git add SomeFileNameYouChanged

on the modified file(s). This can be a list of filenames!

#### Step 6:

run `git commit` and enter a short message on what you updated. `git` may ask you to identify yourself now. Simply copy-paste what is suggested there and fill in your email and name.

#### Step 7:

Push your changes to __your__ github repository:

    git push mygithub master

Check github.com that something changed there.

#### Step 8:

Create a pull request. Simply click on __New pull request__ in your github account.

Thats it! You are a `git` expert! congratulations!

### Try yourself:

 * try git on our little 'yield_or_return project'




### 'if __name__ == "__main__"'
In python, other .py-files can be imported as modules. Sometimes modules are imported from a file which contains not only function and class definitions, but also executable code. 

When a python script is executed, its __name__ attribute is set to "__main__". If a script is imported to another script, the __name__ is instead set to the name of that script/module.

By doing the "main check", it is ensured, that the executable code is only executed if the module is run as a program itself and not if the module is only imported to use the functions within another script.

A small and simple example:

File 1: adding_numbers.py


    def sum_of_list(list_of_numbers):
        result = 0
        for number in list_of_numbers:
            result += number
        return result

    if __name__ == '__main__':
        print("Test: Sum of list [0,1,2,3,4,5] = ",  sum_of_list(range(6))


Fle 2: myscript.py

    import adding_numbers

    print(adding_numbers.sum_of_list(range(17)))


...if you want to add numbers you would of course not write such a function but use sum or numpy.sum ;-)
