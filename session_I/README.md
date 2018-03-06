Date: 1.3.2018

Topics
------

 Gesa showed us how to use:
 * enumerate in for loops 
 * nameing conventions
 * Flake8
 * list comprehension

 Peter demonstrated:
 * random number generation
 * Numpy broadcasting (working with large arrays in no time)

-------

###Nested for loops

First, we discussed these nested for loops: 

    for trace in range(0,len(mseedfiles)):
        for trace2 in range(0,len(mseedfiles)):
            [ ... ]


`range` always starts at 0, so let's remove the first `0` argument:

    for trace in range(len(mseedfiles)):

For iterating through a list you can also use `enumerate` which returns tuples of (index, object) in each iteration step. This comes handy. Use thoughtfully chosen names. E.g. start an index name with an `i`. Avoid naming objects like modules (`trace` is a module in `pyrocko`!).
A better solution would be:

    for ia, atrace in enumerate(mseedfiles):
        [ ... ]


`ia` is the index of `atrace` in the list `mseedfiles`. This naming has the advantage that nested loops can be defined in a more systematic fashion:

    for ia, atrace in enumerate(mseedfiles):
        for ib, btrace in enumerate(mseedfiles):
            [ ... ]

###Style conventions

Another line we discussed:  

    tr.filter("bandpass",freqmin=bandpass[0],freqmax=bandpass[1],zerophase=True) 

You can use [flake8](http://flake8.pycqa.org/en/latest/) to check the style. This is what `flake8` tells us about this single line:


    test.py:1:29: E231 missing whitespace after ','
    test.py:1:49: E231 missing whitespace after ','
    test.py:1:69: E231 missing whitespace after ','
    test.py:1:80: E501 line too long (84 > 79 characters)

As you can see, PEP8 suggest to add whitespaces after each ',' and to keep lines shorter than 80 characters. You can use line breaks after each comma instead:

    tr.filter("bandpass",   
              freqmin=bandpass[0],
              freqmax=bandpass[1],
              zerophase=True)

Nice!

###List comprehension

Finally, we discussed [list comprehension](http://www.pythonforbeginners.com/basics/list-comprehensions-in-python). In order to extract station codes from a list of stations you could do:

    station_codes = []
    for stat in stations:
        station_codes.append(stat.nsl())

This does to job. But there is a more 'pythonic' way of doing it (which can be up to 20% faster), called list comprehension:

    station_codes = [stat.nsl() for stat in station]

It's not only more efficient but also saves 2 out of 3 lines of code which is good - your code becomes more easy to read!

###Random number generation and numpy broadcasting

First let's have a look onto the ugly approach for calculating simple averages of random values:

```
import numpy as np
import random, time

start_time = time.time()

output = np.zeros((len(val_list),len(val_list)))

cat_list = range(len(val_list))

# creating a list using python's standard random is very slow 
val_list = [random.randrange(1,101,1) for x in range (25000)]

# bad bad nested loop
for i in cat_list:
    for j in cat_list:
            output[i,j] = (val_list[i] + val_list[j])/2.

print 'Time: %.6f' % (time.time() - start_time)
```

For random number generation we switch to a much faster numpy function with only slightly changes in syntax
```
values = np.random.randint(1,101,25000)
```

... and instead of iterating over all values twice we will use numpy's matrix broadcasting:

At first we convert the values given as a row vector into a column vector. This is done by adding a new axis:
```
values[:,np.newaxis]
```

Now we add up both vectors and devide all values by 2:
```
output = (values + values[:,np.newaxis])/2.
```
Wait! Did we just add a column vector to a row vector? Yes, we did. Though, in theory element-to-element operations are not defined if two matrixes have different shapes. However, numpy does it anyway by using broadcasting. Broadcasting means that both arrays are expanded(row/column-wise copied) until they have compatible shapes.

See this example:
```
          [1]   [[1,2,3]]   [[1,1,1]]   [[2,3,4]]
[1,2,3] + [2] = [[1,2,3]] + [[2,2,2]] = [[3,4,5]]
          [3]   [[1,2,3]]   [[3,3,3]]   [[4,5,6]]
```


Putting the above stuff into the new code:
```
import numpy as np
import time

start_time = time.time()

values = np.random.randint(1,101,25)

output = (values + values[:,np.newaxis])/2.

print 'Time: %.6f' % (time.time() - start_time)
```  

By two small changes we made our example code running about 25 times faster.

###Suggested Homework:
 * install `flake8`. Might work like this:

    python -m pip install flake8

 * run `flake8` on one of your scripts and correct it
 * Check your codes and replace a short for-loop with a list comprehension.

Tips about documenting your code: http://docs.python-guide.org/en/latest/writing/documentation/
