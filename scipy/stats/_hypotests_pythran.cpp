#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/include/types/float.hpp>
#include <pythonic/include/types/numpy_texpr.hpp>
#include <pythonic/include/types/int.hpp>
#include <pythonic/types/float.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/int.hpp>
#include <pythonic/types/numpy_texpr.hpp>
#include <pythonic/include/builtins/getattr.hpp>
#include <pythonic/include/builtins/range.hpp>
#include <pythonic/include/builtins/tuple.hpp>
#include <pythonic/include/numpy/square.hpp>
#include <pythonic/include/numpy/sum.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/include/operator_/iadd.hpp>
#include <pythonic/include/operator_/mul.hpp>
#include <pythonic/include/operator_/sub.hpp>
#include <pythonic/include/types/slice.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/builtins/getattr.hpp>
#include <pythonic/builtins/range.hpp>
#include <pythonic/builtins/tuple.hpp>
#include <pythonic/numpy/square.hpp>
#include <pythonic/numpy/sum.hpp>
#include <pythonic/operator_/add.hpp>
#include <pythonic/operator_/iadd.hpp>
#include <pythonic/operator_/mul.hpp>
#include <pythonic/operator_/sub.hpp>
#include <pythonic/types/slice.hpp>
#include <pythonic/types/str.hpp>
namespace __pythran__hypotests_pythran
{
  struct _a_ij_Aij_Dij2
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef typename pythonic::assignable<long>::type __type2;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type4;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type0>())) __type6;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type6>::type>::type __type7;
      typedef typename pythonic::lazy<__type7>::type __type8;
      typedef decltype(std::declval<__type4>()(std::declval<__type8>())) __type9;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type9>::type::iterator>::value_type>::type __type10;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type6>::type>::type __type11;
      typedef typename pythonic::lazy<__type11>::type __type12;
      typedef decltype(std::declval<__type4>()(std::declval<__type12>())) __type13;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type13>::type::iterator>::value_type>::type __type14;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type10>(), std::declval<__type14>())) __type15;
      typedef decltype(std::declval<__type0>()[std::declval<__type15>()]) __type16;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::square{})>::type>::type __type17;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type18;
      typedef typename pythonic::assignable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type __type19;
      typedef typename __combined<__type19,__type0>::type __type20;
      typedef pythonic::types::contiguous_slice __type21;
      typedef decltype(std::declval<__type20>()(std::declval<__type21>(), std::declval<__type21>())) __type22;
      typedef decltype(std::declval<__type18>()(std::declval<__type22>())) __type23;
      typedef decltype(pythonic::operator_::add(std::declval<__type23>(), std::declval<__type23>())) __type26;
      typedef typename pythonic::assignable<typename __combined<__type0,__type0>::type>::type __type27;
      typedef decltype(std::declval<__type27>()(std::declval<__type21>(), std::declval<__type21>())) __type28;
      typedef decltype(std::declval<__type18>()(std::declval<__type28>())) __type29;
      typedef decltype(pythonic::operator_::add(std::declval<__type29>(), std::declval<__type29>())) __type32;
      typedef decltype(pythonic::operator_::sub(std::declval<__type26>(), std::declval<__type32>())) __type33;
      typedef decltype(std::declval<__type17>()(std::declval<__type33>())) __type34;
      typedef decltype(pythonic::operator_::mul(std::declval<__type16>(), std::declval<__type34>())) __type35;
      typedef decltype(pythonic::operator_::add(std::declval<__type2>(), std::declval<__type35>())) __type36;
      typedef typename __combined<__type2,__type36>::type __type37;
      typedef __type0 __ptype0;
      typedef __type0 __ptype1;
      typedef typename pythonic::returnable<typename __combined<__type37,__type35>::type>::type result_type;
    }  
    ;
    template <typename argument_type0 >
    typename type<argument_type0>::result_type operator()(argument_type0&& A) const
    ;
  }  ;
  struct _Q
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef typename pythonic::assignable<long>::type __type1;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type3;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type0>())) __type5;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type5>::type>::type __type6;
      typedef typename pythonic::lazy<__type6>::type __type7;
      typedef decltype(std::declval<__type3>()(std::declval<__type7>())) __type8;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type8>::type::iterator>::value_type>::type __type9;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type5>::type>::type __type10;
      typedef typename pythonic::lazy<__type10>::type __type11;
      typedef decltype(std::declval<__type3>()(std::declval<__type11>())) __type12;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type12>::type::iterator>::value_type>::type __type13;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type9>(), std::declval<__type13>())) __type14;
      typedef decltype(std::declval<__type0>()[std::declval<__type14>()]) __type15;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type16;
      typedef typename pythonic::assignable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type __type17;
      typedef pythonic::types::contiguous_slice __type18;
      typedef decltype(std::declval<__type17>()(std::declval<__type18>(), std::declval<__type18>())) __type19;
      typedef decltype(std::declval<__type16>()(std::declval<__type19>())) __type20;
      typedef decltype(pythonic::operator_::add(std::declval<__type20>(), std::declval<__type20>())) __type23;
      typedef decltype(pythonic::operator_::mul(std::declval<__type15>(), std::declval<__type23>())) __type24;
      typedef decltype(pythonic::operator_::add(std::declval<__type1>(), std::declval<__type24>())) __type25;
      typedef typename __combined<__type1,__type25>::type __type26;
      typedef __type0 __ptype2;
      typedef typename pythonic::returnable<typename __combined<__type26,__type24>::type>::type result_type;
    }  
    ;
    template <typename argument_type0 >
    typename type<argument_type0>::result_type operator()(argument_type0&& A) const
    ;
  }  ;
  struct _P
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef typename pythonic::assignable<long>::type __type1;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type3;
      typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type0>())) __type5;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type5>::type>::type __type6;
      typedef typename pythonic::lazy<__type6>::type __type7;
      typedef decltype(std::declval<__type3>()(std::declval<__type7>())) __type8;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type8>::type::iterator>::value_type>::type __type9;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type5>::type>::type __type10;
      typedef typename pythonic::lazy<__type10>::type __type11;
      typedef decltype(std::declval<__type3>()(std::declval<__type11>())) __type12;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type12>::type::iterator>::value_type>::type __type13;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type9>(), std::declval<__type13>())) __type14;
      typedef decltype(std::declval<__type0>()[std::declval<__type14>()]) __type15;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type16;
      typedef typename pythonic::assignable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type __type17;
      typedef pythonic::types::contiguous_slice __type18;
      typedef decltype(std::declval<__type17>()(std::declval<__type18>(), std::declval<__type18>())) __type19;
      typedef decltype(std::declval<__type16>()(std::declval<__type19>())) __type20;
      typedef decltype(pythonic::operator_::add(std::declval<__type20>(), std::declval<__type20>())) __type23;
      typedef decltype(pythonic::operator_::mul(std::declval<__type15>(), std::declval<__type23>())) __type24;
      typedef decltype(pythonic::operator_::add(std::declval<__type1>(), std::declval<__type24>())) __type25;
      typedef typename __combined<__type1,__type25>::type __type26;
      typedef __type0 __ptype3;
      typedef typename pythonic::returnable<typename __combined<__type26,__type24>::type>::type result_type;
    }  
    ;
    template <typename argument_type0 >
    typename type<argument_type0>::result_type operator()(argument_type0&& A) const
    ;
  }  ;
  struct _Dij
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type1;
      typedef pythonic::types::contiguous_slice __type2;
      typedef decltype(std::declval<__type1>()(std::declval<__type2>(), std::declval<__type2>())) __type3;
      typedef decltype(std::declval<__type0>()(std::declval<__type3>())) __type4;
      typedef typename pythonic::returnable<decltype(pythonic::operator_::add(std::declval<__type4>(), std::declval<__type4>()))>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    typename type<argument_type0, argument_type1, argument_type2>::result_type operator()(argument_type0&& A, argument_type1&& i, argument_type2&& j) const
    ;
  }  ;
  struct _Aij
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type1;
      typedef pythonic::types::contiguous_slice __type2;
      typedef decltype(std::declval<__type1>()(std::declval<__type2>(), std::declval<__type2>())) __type3;
      typedef decltype(std::declval<__type0>()(std::declval<__type3>())) __type4;
      typedef typename pythonic::returnable<decltype(pythonic::operator_::add(std::declval<__type4>(), std::declval<__type4>()))>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    typename type<argument_type0, argument_type1, argument_type2>::result_type operator()(argument_type0&& A, argument_type1&& i, argument_type2&& j) const
    ;
  }  ;
  template <typename argument_type0 >
  typename _a_ij_Aij_Dij2::type<argument_type0>::result_type _a_ij_Aij_Dij2::operator()(argument_type0&& A) const
  {
    typedef typename pythonic::assignable<long>::type __type0;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type1;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type2;
    typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type1>())) __type4;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type4>::type>::type __type5;
    typedef typename pythonic::lazy<__type5>::type __type6;
    typedef decltype(std::declval<__type2>()(std::declval<__type6>())) __type7;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type __type8;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type4>::type>::type __type9;
    typedef typename pythonic::lazy<__type9>::type __type10;
    typedef decltype(std::declval<__type2>()(std::declval<__type10>())) __type11;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type11>::type::iterator>::value_type>::type __type12;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type8>(), std::declval<__type12>())) __type13;
    typedef decltype(std::declval<__type1>()[std::declval<__type13>()]) __type14;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::square{})>::type>::type __type15;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type16;
    typedef typename pythonic::assignable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type __type17;
    typedef typename __combined<__type17,__type1>::type __type19;
    typedef pythonic::types::contiguous_slice __type20;
    typedef decltype(std::declval<__type19>()(std::declval<__type20>(), std::declval<__type20>())) __type21;
    typedef decltype(std::declval<__type16>()(std::declval<__type21>())) __type22;
    typedef decltype(pythonic::operator_::add(std::declval<__type22>(), std::declval<__type22>())) __type25;
    typedef typename pythonic::assignable<typename __combined<__type1,__type1>::type>::type __type27;
    typedef decltype(std::declval<__type27>()(std::declval<__type20>(), std::declval<__type20>())) __type28;
    typedef decltype(std::declval<__type16>()(std::declval<__type28>())) __type29;
    typedef decltype(pythonic::operator_::add(std::declval<__type29>(), std::declval<__type29>())) __type32;
    typedef decltype(pythonic::operator_::sub(std::declval<__type25>(), std::declval<__type32>())) __type33;
    typedef decltype(std::declval<__type15>()(std::declval<__type33>())) __type34;
    typedef decltype(pythonic::operator_::mul(std::declval<__type14>(), std::declval<__type34>())) __type35;
    typedef decltype(pythonic::operator_::add(std::declval<__type0>(), std::declval<__type35>())) __type36;
    typedef typename __combined<__type0,__type36>::type __type37;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type>::type i;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type11>::type::iterator>::value_type>::type>::type j;
    typename pythonic::lazy<decltype(std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A)))>::type m = std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A));
    typename pythonic::lazy<decltype(std::get<1>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A)))>::type n = std::get<1>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A));
    typename pythonic::assignable<typename __combined<__type37,__type35>::type>::type count = 0L;
    {
      long  __target140551422452112 = m;
      for (long  i=0L; i < __target140551422452112; i += 1L)
      {
        {
          long  __target140551422454176 = n;
          for (long  j=0L; j < __target140551422454176; j += 1L)
          {
            typename pythonic::assignable<typename __combined<__type17,__type1>::type>::type __pythran_inline_AijA2 = A;
            typename pythonic::assignable_noescape<decltype(i)>::type __pythran_inline_Aiji2 = i;
            typename pythonic::assignable_noescape<decltype(j)>::type __pythran_inline_Aijj2 = j;
            typename pythonic::assignable<typename pythonic::assignable<typename __combined<__type1,__type1>::type>::type>::type __pythran_inline_DijA3 = A;
            typename pythonic::assignable_noescape<decltype(i)>::type __pythran_inline_Diji3 = i;
            typename pythonic::assignable_noescape<decltype(j)>::type __pythran_inline_Dijj3 = j;
            count += pythonic::operator_::mul(A.fast(pythonic::types::make_tuple(i, j)), pythonic::numpy::functor::square{}(pythonic::operator_::sub(pythonic::operator_::add(pythonic::numpy::functor::sum{}(__pythran_inline_AijA2(pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Aiji2),pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Aijj2))), pythonic::numpy::functor::sum{}(__pythran_inline_AijA2(pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Aiji2, 1L),pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Aijj2, 1L),pythonic::builtins::None)))), pythonic::operator_::add(pythonic::numpy::functor::sum{}(__pythran_inline_DijA3(pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Diji3, 1L),pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Dijj3))), pythonic::numpy::functor::sum{}(__pythran_inline_DijA3(pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Diji3),pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Dijj3, 1L),pythonic::builtins::None)))))));
          }
        }
      }
    }
    return count;
  }
  template <typename argument_type0 >
  typename _Q::type<argument_type0>::result_type _Q::operator()(argument_type0&& A) const
  {
    typedef typename pythonic::assignable<long>::type __type0;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type1;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type2;
    typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type1>())) __type4;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type4>::type>::type __type5;
    typedef typename pythonic::lazy<__type5>::type __type6;
    typedef decltype(std::declval<__type2>()(std::declval<__type6>())) __type7;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type __type8;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type4>::type>::type __type9;
    typedef typename pythonic::lazy<__type9>::type __type10;
    typedef decltype(std::declval<__type2>()(std::declval<__type10>())) __type11;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type11>::type::iterator>::value_type>::type __type12;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type8>(), std::declval<__type12>())) __type13;
    typedef decltype(std::declval<__type1>()[std::declval<__type13>()]) __type14;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type15;
    typedef typename pythonic::assignable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type __type16;
    typedef pythonic::types::contiguous_slice __type17;
    typedef decltype(std::declval<__type16>()(std::declval<__type17>(), std::declval<__type17>())) __type18;
    typedef decltype(std::declval<__type15>()(std::declval<__type18>())) __type19;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type19>())) __type22;
    typedef decltype(pythonic::operator_::mul(std::declval<__type14>(), std::declval<__type22>())) __type23;
    typedef decltype(pythonic::operator_::add(std::declval<__type0>(), std::declval<__type23>())) __type24;
    typedef typename __combined<__type0,__type24>::type __type25;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type>::type i;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type11>::type::iterator>::value_type>::type>::type j;
    typename pythonic::lazy<decltype(std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A)))>::type m = std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A));
    typename pythonic::lazy<decltype(std::get<1>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A)))>::type n = std::get<1>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A));
    typename pythonic::assignable<typename __combined<__type25,__type23>::type>::type count = 0L;
    {
      long  __target140551422447680 = m;
      for (long  i=0L; i < __target140551422447680; i += 1L)
      {
        {
          long  __target140551422449744 = n;
          for (long  j=0L; j < __target140551422449744; j += 1L)
          {
            typename pythonic::assignable_noescape<decltype(A)>::type __pythran_inline_DijA1 = A;
            typename pythonic::assignable_noescape<decltype(i)>::type __pythran_inline_Diji1 = i;
            typename pythonic::assignable_noescape<decltype(j)>::type __pythran_inline_Dijj1 = j;
            count += pythonic::operator_::mul(A.fast(pythonic::types::make_tuple(i, j)), pythonic::operator_::add(pythonic::numpy::functor::sum{}(__pythran_inline_DijA1(pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Diji1, 1L),pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Dijj1))), pythonic::numpy::functor::sum{}(__pythran_inline_DijA1(pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Diji1),pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Dijj1, 1L),pythonic::builtins::None)))));
          }
        }
      }
    }
    return count;
  }
  template <typename argument_type0 >
  typename _P::type<argument_type0>::result_type _P::operator()(argument_type0&& A) const
  {
    typedef typename pythonic::assignable<long>::type __type0;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type1;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type2;
    typedef decltype(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, std::declval<__type1>())) __type4;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type4>::type>::type __type5;
    typedef typename pythonic::lazy<__type5>::type __type6;
    typedef decltype(std::declval<__type2>()(std::declval<__type6>())) __type7;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type __type8;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type4>::type>::type __type9;
    typedef typename pythonic::lazy<__type9>::type __type10;
    typedef decltype(std::declval<__type2>()(std::declval<__type10>())) __type11;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type11>::type::iterator>::value_type>::type __type12;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type8>(), std::declval<__type12>())) __type13;
    typedef decltype(std::declval<__type1>()[std::declval<__type13>()]) __type14;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::sum{})>::type>::type __type15;
    typedef typename pythonic::assignable<typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type>::type __type16;
    typedef pythonic::types::contiguous_slice __type17;
    typedef decltype(std::declval<__type16>()(std::declval<__type17>(), std::declval<__type17>())) __type18;
    typedef decltype(std::declval<__type15>()(std::declval<__type18>())) __type19;
    typedef decltype(pythonic::operator_::add(std::declval<__type19>(), std::declval<__type19>())) __type22;
    typedef decltype(pythonic::operator_::mul(std::declval<__type14>(), std::declval<__type22>())) __type23;
    typedef decltype(pythonic::operator_::add(std::declval<__type0>(), std::declval<__type23>())) __type24;
    typedef typename __combined<__type0,__type24>::type __type25;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type7>::type::iterator>::value_type>::type>::type i;
    typename pythonic::assignable<typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type11>::type::iterator>::value_type>::type>::type j;
    typename pythonic::lazy<decltype(std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A)))>::type m = std::get<0>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A));
    typename pythonic::lazy<decltype(std::get<1>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A)))>::type n = std::get<1>(pythonic::builtins::getattr(pythonic::types::attr::SHAPE{}, A));
    typename pythonic::assignable<typename __combined<__type25,__type23>::type>::type count = 0L;
    {
      long  __target140551422460016 = m;
      for (long  i=0L; i < __target140551422460016; i += 1L)
      {
        {
          long  __target140551422482704 = n;
          for (long  j=0L; j < __target140551422482704; j += 1L)
          {
            typename pythonic::assignable_noescape<decltype(A)>::type __pythran_inline_AijA0 = A;
            typename pythonic::assignable_noescape<decltype(i)>::type __pythran_inline_Aiji0 = i;
            typename pythonic::assignable_noescape<decltype(j)>::type __pythran_inline_Aijj0 = j;
            count += pythonic::operator_::mul(A.fast(pythonic::types::make_tuple(i, j)), pythonic::operator_::add(pythonic::numpy::functor::sum{}(__pythran_inline_AijA0(pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Aiji0),pythonic::types::contiguous_slice(pythonic::builtins::None,__pythran_inline_Aijj0))), pythonic::numpy::functor::sum{}(__pythran_inline_AijA0(pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Aiji0, 1L),pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::operator_::add(__pythran_inline_Aijj0, 1L),pythonic::builtins::None)))));
          }
        }
      }
    }
    return count;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
  typename _Dij::type<argument_type0, argument_type1, argument_type2>::result_type _Dij::operator()(argument_type0&& A, argument_type1&& i, argument_type2&& j) const
  {
    return pythonic::operator_::add(pythonic::numpy::functor::sum{}(A(pythonic::types::contiguous_slice(pythonic::operator_::add(i, 1L),pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,j))), pythonic::numpy::functor::sum{}(A(pythonic::types::contiguous_slice(pythonic::builtins::None,i),pythonic::types::contiguous_slice(pythonic::operator_::add(j, 1L),pythonic::builtins::None))));
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
  typename _Aij::type<argument_type0, argument_type1, argument_type2>::result_type _Aij::operator()(argument_type0&& A, argument_type1&& i, argument_type2&& j) const
  {
    return pythonic::operator_::add(pythonic::numpy::functor::sum{}(A(pythonic::types::contiguous_slice(pythonic::builtins::None,i),pythonic::types::contiguous_slice(pythonic::builtins::None,j))), pythonic::numpy::functor::sum{}(A(pythonic::types::contiguous_slice(pythonic::operator_::add(i, 1L),pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::operator_::add(j, 1L),pythonic::builtins::None))));
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
typename __pythran__hypotests_pythran::_a_ij_Aij_Dij2::type<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>::result_type _a_ij_Aij_Dij20(pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_a_ij_Aij_Dij2()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_a_ij_Aij_Dij2::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>::result_type _a_ij_Aij_Dij21(pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_a_ij_Aij_Dij2()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_a_ij_Aij_Dij2::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>::result_type _a_ij_Aij_Dij22(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_a_ij_Aij_Dij2()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_a_ij_Aij_Dij2::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>::result_type _a_ij_Aij_Dij23(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_a_ij_Aij_Dij2()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Q::type<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>::result_type _Q0(pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Q()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Q::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>::result_type _Q1(pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Q()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Q::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>::result_type _Q2(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Q()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Q::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>::result_type _Q3(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Q()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_P::type<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>::result_type _P0(pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_P()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_P::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>::result_type _P1(pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_P()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_P::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>::result_type _P2(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_P()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_P::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>::result_type _P3(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& A) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_P()(A);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Dij::type<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>, long, long>::result_type _Dij0(pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Dij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Dij::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>, long, long>::result_type _Dij1(pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Dij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Dij::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, long, long>::result_type _Dij2(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Dij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Dij::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, long, long>::result_type _Dij3(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Dij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Aij::type<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>, long, long>::result_type _Aij0(pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Aij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Aij::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>, long, long>::result_type _Aij1(pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Aij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Aij::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, long, long>::result_type _Aij2(pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Aij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__hypotests_pythran::_Aij::type<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, long, long>::result_type _Aij3(pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& A, long&& i, long&& j) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__hypotests_pythran::_Aij()(A, i, j);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap__a_ij_Aij_Dij20(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]))
        return to_python(_a_ij_Aij_Dij20(from_python<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__a_ij_Aij_Dij21(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]))
        return to_python(_a_ij_Aij_Dij21(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__a_ij_Aij_Dij22(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]))
        return to_python(_a_ij_Aij_Dij22(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__a_ij_Aij_Dij23(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]))
        return to_python(_a_ij_Aij_Dij23(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Q0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]))
        return to_python(_Q0(from_python<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Q1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]))
        return to_python(_Q1(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Q2(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]))
        return to_python(_Q2(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Q3(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]))
        return to_python(_Q3(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__P0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]))
        return to_python(_P0(from_python<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__P1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]))
        return to_python(_P1(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__P2(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]))
        return to_python(_P2(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__P3(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];
    char const* keywords[] = {"A",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]))
        return to_python(_P3(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Dij0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Dij0(from_python<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Dij1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Dij1(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Dij2(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Dij2(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Dij3(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Dij3(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Aij0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Aij0(from_python<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Aij1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Aij1(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<long,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Aij2(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Aij2(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap__Aij3(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    char const* keywords[] = {"A", "i", "j",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]) && is_convertible<long>(args_obj[1]) && is_convertible<long>(args_obj[2]))
        return to_python(_Aij3(from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[0]), from_python<long>(args_obj[1]), from_python<long>(args_obj[2])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall__a_ij_Aij_Dij2(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__a_ij_Aij_Dij20(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__a_ij_Aij_Dij21(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__a_ij_Aij_Dij22(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__a_ij_Aij_Dij23(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_a_ij_Aij_Dij2", "\n""    - _a_ij_Aij_Dij2(int[:,:])\n""    - _a_ij_Aij_Dij2(float[:,:])", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall__Q(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__Q0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Q1(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Q2(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Q3(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_Q", "\n""    - _Q(int[:,:])\n""    - _Q(float[:,:])", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall__P(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__P0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__P1(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__P2(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__P3(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_P", "\n""    - _P(int[:,:])\n""    - _P(float[:,:])", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall__Dij(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__Dij0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Dij1(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Dij2(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Dij3(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_Dij", "\n""    - _Dij(int[:,:], int, int)\n""    - _Dij(float[:,:], int, int)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall__Aij(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap__Aij0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Aij1(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Aij2(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap__Aij3(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "_Aij", "\n""    - _Aij(int[:,:], int, int)\n""    - _Aij(float[:,:], int, int)", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "_a_ij_Aij_Dij2",
    (PyCFunction)__pythran_wrapall__a_ij_Aij_Dij2,
    METH_VARARGS | METH_KEYWORDS,
    "A term that appears in the ASE of Kendall's tau and Somers' D.\n""\n""    Supported prototypes:\n""\n""    - _a_ij_Aij_Dij2(int[:,:])\n""    - _a_ij_Aij_Dij2(float[:,:])"},{
    "_Q",
    (PyCFunction)__pythran_wrapall__Q,
    METH_VARARGS | METH_KEYWORDS,
    "Twice the number of discordant pairs, excluding ties.\n""\n""    Supported prototypes:\n""\n""    - _Q(int[:,:])\n""    - _Q(float[:,:])"},{
    "_P",
    (PyCFunction)__pythran_wrapall__P,
    METH_VARARGS | METH_KEYWORDS,
    "Twice the number of concordant pairs, excluding ties.\n""\n""    Supported prototypes:\n""\n""    - _P(int[:,:])\n""    - _P(float[:,:])"},{
    "_Dij",
    (PyCFunction)__pythran_wrapall__Dij,
    METH_VARARGS | METH_KEYWORDS,
    "Sum of lower-left and upper-right blocks of contingency table.\n""\n""    Supported prototypes:\n""\n""    - _Dij(int[:,:], int, int)\n""    - _Dij(float[:,:], int, int)"},{
    "_Aij",
    (PyCFunction)__pythran_wrapall__Aij,
    METH_VARARGS | METH_KEYWORDS,
    "Sum of upper-left and lower right blocks of contingency table.\n""\n""    Supported prototypes:\n""\n""    - _Aij(int[:,:], int, int)\n""    - _Aij(float[:,:], int, int)"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_hypotests_pythran",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_hypotests_pythran)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
__attribute__ ((externally_visible))
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_hypotests_pythran)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("_hypotests_pythran",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(sss)",
                                      "0.9.11",
                                      "2021-07-15 09:57:59.527404",
                                      "c1daeef739e190b001bcd9fbabc6b217f8584808fd9822565cbb8dfefdc46c11");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif