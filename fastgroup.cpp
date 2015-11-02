/**
 * Fastgroup algorithm
 * implemented in C++
 * @author Patrick Kage
 */


// this requires that python's install location is visible to your compiler. For OSX, this is
// inside /Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 .
#include <Python.h>
#include <iostream>

// cuz we're lazy
using namespace std;

// i mean i could use prototypes here but i'm not going to do that



// test method, tests a method
static PyObject * test(PyObject *self, PyObject *args) {
	cout << "test";
	Py_INCREF(Py_None);
	return Py_None;
}

// the group algorithm itself!
static PyObject * group(PyObject *self, PyObject *args) {
	cout << "fastgrouping...";
	Py_INCREF(Py_None);
	return Py_None;
}


// module table definition - metadata thingamajig
static PyMethodDef FastgroupMethods[] = {
	{
		"group", group, METH_VARARGS,
		"find all connected pixels in a group"
	},
	{
		"test", test, METH_VARARGS,
		"a test function"
	},
	{ // sentinel function? i guess? recommended by docs
		NULL, NULL, 0, NULL
	} 
};

// initializer
PyMODINIT_FUNC initfastgroup() {
	(void) Py_InitModule("fastgroup", FastgroupMethods); // whee!
}



int main(int argc, char** argv) {
	// set the program name
	Py_SetProgramName(argv[0]);

	// init the python interpreter
	Py_Initialize();

	// init our module
	initfastgroup();

};
