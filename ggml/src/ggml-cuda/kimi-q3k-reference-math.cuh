// Experimental CUDA translation of glibc 2.35/fdlibm float routines.
// Original notices follow; glibc expf/table are LGPL-2.1-or-later.
// Strict RN, FTZ off, errno/fenv side effects are not part of this API.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#define GET_FLOAT_WORD(i,d) ((i)=__float_as_uint(d))
#define SET_FLOAT_WORD(d,i) ((d)=__uint_as_float(uint32_t(i)))
#define math_check_force_underflow(x) ((void)0)
#define math_force_eval(x) ((void)0)
#define __set_errno(x) ((void)0)
#define ERANGE 34
namespace reference_expm1 {
/* s_expm1f.c -- float version of s_expm1.c.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


static constexpr float huge = 1.0e+30;
static constexpr float tiny = 1.0e-30;

static constexpr float
one		= 1.0,
o_threshold	= 8.8721679688e+01,/* 0x42b17180 */
ln2_hi		= 6.9313812256e-01,/* 0x3f317180 */
ln2_lo		= 9.0580006145e-06,/* 0x3717f7d1 */
invln2		= 1.4426950216e+00,/* 0x3fb8aa3b */
	/* scaled coefficients related to expm1 */
Q1  =  -3.3333335072e-02, /* 0xbd088889 */
Q2  =   1.5873016091e-03, /* 0x3ad00d01 */
Q3  =  -7.9365076090e-05, /* 0xb8a670cd */
Q4  =   4.0082177293e-06, /* 0x36867e54 */
Q5  =  -2.0109921195e-07; /* 0xb457edbb */

__device__ float __expm1f(float x)
{
	float y,hi,lo,c,t,e,hxs,hfx,r1;
	int32_t k,xsb;
	uint32_t hx;

	GET_FLOAT_WORD(hx,x);
	xsb = hx&0x80000000;		/* sign bit of x */
	if(xsb==0) y=x; else y= -x;	/* y = |x| */
	hx &= 0x7fffffff;		/* high word of |x| */

    /* filter out huge and non-finite argument */
	if(hx >= 0x4195b844) {			/* if |x|>=27*ln2 */
	    if(hx >= 0x42b17218) {		/* if |x|>=88.721... */
		if(hx>0x7f800000)
		    return x+x;		 /* NaN */
		if(hx==0x7f800000)
		    return (xsb==0)? x:-1.0;/* exp(+-inf)={inf,-1} */
		if(x > o_threshold) {
		  __set_errno (ERANGE);
		  return huge*huge; /* overflow */
		}
	    }
	    if(xsb!=0) { /* x < -27*ln2, return -1.0 with inexact */
		math_force_eval(x+tiny);/* raise inexact */
		return tiny-one;	/* return -1 */
	    }
	}

    /* argument reduction */
	if(hx > 0x3eb17218) {		/* if  |x| > 0.5 ln2 */
	    if(hx < 0x3F851592) {	/* and |x| < 1.5 ln2 */
		if(xsb==0)
		    {hi = x - ln2_hi; lo =  ln2_lo;  k =  1;}
		else
		    {hi = x + ln2_hi; lo = -ln2_lo;  k = -1;}
	    } else {
		k  = invln2*x+((xsb==0)?(float)0.5:(float)-0.5);
		t  = k;
		hi = x - t*ln2_hi;	/* t*ln2_hi is exact here */
		lo = t*ln2_lo;
	    }
	    x  = hi - lo;
	    c  = (hi-x)-lo;
	}
	else if(hx < 0x33000000) {	/* when |x|<2**-25, return x */
	    math_check_force_underflow (x);
	    t = huge+x;	/* return x with inexact flags when x!=0 */
	    return x - (t-(huge+x));
	}
	else k = 0;

    /* x is now in primary range */
	hfx = (float)0.5*x;
	hxs = x*hfx;
	r1 = one+hxs*(Q1+hxs*(Q2+hxs*(Q3+hxs*(Q4+hxs*Q5))));
	t  = (float)3.0-r1*hfx;
	e  = hxs*((r1-t)/((float)6.0 - x*t));
	if(k==0) return x - (x*e-hxs);		/* c is 0 */
	else {
	    e  = (x*(e-c)-c);
	    e -= hxs;
	    if(k== -1) return (float)0.5*(x-e)-(float)0.5;
	    if(k==1) {
		if(x < (float)-0.25) return -(float)2.0*(e-(x+(float)0.5));
		else	      return  one+(float)2.0*(x-e);
	    }
	    if (k <= -2 || k>56) {   /* suffice to return exp(x)-1 */
		int32_t i;
		y = one-(e-x);
		GET_FLOAT_WORD(i,y);
		SET_FLOAT_WORD(y,i+(uint32_t(k)<<23));	/* add k to y's exponent */
		return y-one;
	    }
	    t = one;
	    if(k<23) {
		int32_t i;
		SET_FLOAT_WORD(t,0x3f800000 - (0x1000000>>k)); /* t=1-2^-k */
		y = t-(e-x);
		GET_FLOAT_WORD(i,y);
		SET_FLOAT_WORD(y,i+(uint32_t(k)<<23));	/* add k to y's exponent */
	   } else {
		int32_t i;
		SET_FLOAT_WORD(t,((0x7f-k)<<23));	/* 2^-k */
		y = x-(e+t);
		y += one;
		GET_FLOAT_WORD(i,y);
		SET_FLOAT_WORD(y,i+(uint32_t(k)<<23));	/* add k to y's exponent */
	    }
	}
	return y;
}


}
namespace reference_tanh {
/* s_tanhf.c -- float version of s_tanh.c.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_tanhf.c,v 1.4 1995/05/10 20:48:24 jtc Exp $";
#endif


static constexpr float one=1.0, two=2.0, tiny = 1.0e-30;

__device__ float __tanhf(float x)
{
	float t,z;
	int32_t jx,ix;

	GET_FLOAT_WORD(jx,x);
	ix = jx&0x7fffffff;

    /* x is INF or NaN */
	if(ix>=0x7f800000) {
	    if (jx>=0) return one/x+one;    /* tanh(+-inf)=+-1 */
	    else       return one/x-one;    /* tanh(NaN) = NaN */
	}

    /* |x| < 22 */
	if (ix < 0x41b00000) {		/* |x|<22 */
	    if (ix == 0)
		return x;		/* x == +-0 */
	    if (ix<0x24000000) 		/* |x|<2**-55 */
	      {
		math_check_force_underflow (x);
		return x*(one+x);    	/* tanh(small) = small */
	      }
	    if (ix>=0x3f800000) {	/* |x|>=1  */
		t = reference_expm1::__expm1f(two*fabsf(x));
		z = one - two/(t+two);
	    } else {
	        t = reference_expm1::__expm1f(-two*fabsf(x));
	        z= -t/(t+two);
	    }
    /* |x| > 22, return +-1 */
	} else {
	    z = one - tiny;		/* raised inexact flag */
	}
	return (jx>=0)? z: -z;
}


}
/* Shared data between expf, exp2f and powf.
   Copyright (C) 2017-2022 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */


__device__ __constant__ uint64_t exp_table[32]={
0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
  };
// Explicit FMA sequence from resolved __expf_fma in the frozen B libm.
__device__ float reference_exp(float x) {
    const uint32_t bits=__float_as_uint(x), ab=bits&0x7fffffff;
    if(bits==0xff800000u)return 0.0f;
    if(ab>=0x7f800000u)return x+x;
    if(x>0x1.62e42ep6f)return __uint_as_float(0x7f800000u);
    if(x<-0x1.9fe368p6f)return 0.0f;
    const double inv=0x1.71547652b82fep+0*32,shift=0x1.8p+52;
    double kd=__fma_rn(inv,double(x),shift);
    const uint64_t ki=__double_as_longlong(kd);
    kd=__dsub_rn(kd,shift);
    const double r=__fma_rn(inv,double(x),-kd);
    const double scale=__longlong_as_double(exp_table[ki&31]+(ki<<47));
    const double z=__fma_rn(0x1.c6af84b912394p-5/32/32/32,r,0x1.ebfce50fac4f3p-3/32/32);
    const double y=__fma_rn(z,__dmul_rn(r,r),__fma_rn(0x1.62e42ff0c52d6p-1/32,r,1.0));
    return __double2float_rn(__dmul_rn(y,scale));
}
__device__ float reference_situ(float up,float gate,float beta,float linear) {
    const float a=__fmul_rn(__fmul_rn(beta,reference_tanh::__tanhf(__fmul_rn(gate,__fdiv_rn(1.0f,beta)))),
        __fdiv_rn(1.0f,__fadd_rn(1.0f,reference_exp(-gate))));
    if(linear>0)up=__fmul_rn(linear,reference_tanh::__tanhf(__fmul_rn(up,__fdiv_rn(1.0f,linear))));
    return __fmul_rn(a,up);
}
#undef GET_FLOAT_WORD
#undef SET_FLOAT_WORD
#undef math_check_force_underflow
#undef math_force_eval
#undef __set_errno
#undef ERANGE
