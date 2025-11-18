# gen-sysid

## Description

This project contains code for investigation of genetic algorithms in system identification. In this document all of the rules of contributing are described. Later use cases and manual will be added. In its final form, the code should be available as attachement for any publication resulting from the project.

## Structure of this repository
In the repository, only the code should be stored, any data files should be kept on a cloud drive. The code should be divided into several folders:
+ Data_generation - contains all scripts for generating experimental data
+ Tests - contains all scripts for testing different functions and execution time
+ Tools - contains all scritps with additional tools, like signal processing algorithms
+ Identification - contains all scripts with system identification methods

## How to write the code
This paragraph is a guide, explaining the rules of creating the code for the repository. The following commandments should guide you through code writing procedure:
1. Thou shalt make the comments for thy code in English, to make it international
2. Thou shalt describe the inputs and outputs of thy functions to avoid confusion
3. Thou shalt follow the PEP 8 guidelines
4. Thou shalt write unit tests for thy code
5. Thou shalt make the names of thy variables and functions informative, even if long
6. Thou shalt make thy changes on a separate branch before merging with "master"
7. Thou shalt commit in one message only the changes on one topic

## Usefull sources for Python and Git
+ https://www.python.org/
+ https://www.geeksforgeeks.org/
+ https://realpython.com/
+ https://stackoverflow.com/questions
