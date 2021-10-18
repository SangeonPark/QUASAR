// This file contains a header automatically completed by compilation routines with either the user options or default options.
// It define the template F containing a formula to be intentiate. The formula may be defined in two possible ways: 
//          1) with the user friendly "new syntax"  in FORMULA_OBJ variable with possibly aliases in the variable VAR_ALIASES
//          2) with the machine friendly templated syntax in a variable FORMULA  where the operation are template separated by < >

#pragma once

#define __TYPEACC__ double
#define SUM_SCHEME 1
#define ENABLECHUNK 1


#include <keops_includes.h>

namespace keops {

auto x = Vi(0,3); auto y = Vj(1,3); 

#ifndef USENEWSYNTAX
	#define USENEWSYNTAX 1
#endif

#if USENEWSYNTAX

#define FORMULA_OBJ Sum_Reduction(SqNorm2(x - y),1)
using F = decltype(InvKeopsNS(FORMULA_OBJ));

#else

#define FORMULA @FORMULA@
using F = FORMULA;

#endif
}
