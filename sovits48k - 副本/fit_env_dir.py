import os, re


def fit_activate_bat():
    with open(os.getcwd() + "\\venv\\Scripts\\activate.bat", "r+") as f:
        strings = f.readlines()
        for index, line in enumerate(strings):
            if line.__contains__("VIRTUAL_ENV="):
                res = re.findall('VIRTUAL_ENV=(.*)"', line)[0]
                strings[index] = line.replace(res, os.getcwd() + "\\venv")
    with open(os.getcwd() + "\\venv\\Scripts\\activate.bat", "w+", encoding="utf-8") as h:
        h.writelines(strings)


def fit_Activateps1():
    with open(os.getcwd() + "\\venv\\Scripts\\Activate.ps1", "r+", encoding="utf-8") as e:
        lines = e.readlines()
        for index, line in enumerate(lines):
            if line.__contains__("env:VIRTUAL_ENV="):
                res = re.findall('env:VIRTUAL_ENV="(.*)"', line)[0]
                lines[index] = line.replace(res, os.getcwd() + "\\venv")
    with open(os.getcwd() + "\\venv\\Scripts\\Activate.ps1", "w+", encoding="utf-8") as g:
        g.writelines(lines)


def all_fit():
    fit_activate_bat()
    fit_Activateps1()


if __name__ == '__main__':
    all_fit()