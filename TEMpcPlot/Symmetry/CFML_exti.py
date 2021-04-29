"""
"""
import numpy as np

# zz

eps_ref = 0.0002


Hkl_Ref_Conditions = """ None
(h k l)    h+k=2n : xy0 centred face (C)
(h k l)    k+l=2n : 0yz centred face (A)
(h k l)    h+l=2n : x0z centred face (B)
(h k l)  h+k+l=2n : body centred (I)
(h k l)  h,k,l same parity: all-face centred (F)
(h k l) -h+k+l=3n : rhombohedrally centred (R)
(  0  k  l)     k=2n : (100) glide plane with b/2 translation (b)
(  0  k  l)     l=2n : (100) glide plane with c/2 translation (c)
(  0  k  l)   k+l=2n : (100) glide plane with b/2 + c/2 translations (n)
(  0  k  l)   k+l=4n : (100) glide plane with b/4 +- c/4 translations (d)
(  h  0  l)     h=2n : (010) glide plane with a/2 translation (a)
(  h  0  l)     l=2n : (010) glide plane with c/2 translation (c)
(  h  0  l)   l+h=2n : (010) glide plane with c/2 + a/2 translations (n)
(  h  0  l)   l+h=4n : (010) glide plane with c/4 +- a/4 translations (d)
(  h  k  0)     h=2n : (001) glide plane with a/2 translation (a)
(  h  k  0)     k=2n : (001) glide plane with b/2 translation (b)
(  h  k  0)   h+k=2n : (001) glide plane with a/2 + b/2 translations (n)
(  h  k  0)   h+k=4n : (001) glide plane with a/4 +- b/4 translations (d)
(  h  -h   0 l) l=2n : (11-20) glide plane with c/2 translation (c)
(  0   k  -k l) l=2n : (-2110) glide plane with c/2 translation (c)
( -h   0   h l) l=2n : (1-210) glide plane with c/2 translation (c)
(  h   h -2h l) l=2n : (1-100) glide plane with c/2 translation (c)
(-2h   h   h l) l=2n : (01-10) glide plane with c/2 translation (c)
(  h -2h   h l) l=2n : (-1010) glide plane with c/2 translation (c)
(  h  h  l)     l=2n : (1-10) glide plane with c/2 translation (c,n)
(  h  k  k)     h=2n : (01-1) glide plane with a/2 translation (a,n)
(  h  k  h)     k=2n : (-101) glide plane with b/2 translation (b,n)
(  h  h  l)     l=2n : (1-10) glide plane with c/2 translation (c,n)
(  h  h  l)  2h+l=4n : (1-10) glide plane with a/4 +- b/4 +- c/4 translation (d)
(  h -h  l)     l=2n : (110)  glide plane with c/2 translation (c,n)
(  h -h  l)  2h+l=4n : (110)  glide plane with a/4 +- b/4 +- c/4 translation (d)
(  h  k  k)     h=2n : (01-1) glide plane with a/2 translation (a,n)
(  h  k  k)  2k+h=4n : (01-1) glide plane with +-a/4 + b/4 +- c/4 translation(d)
(  h  k -k)     h=2n : (011)  glide plane with a/2 translation (a,n)
(  h  k -k)  2k+h=4n : (011)  glide plane with +-a/4 + b/4 +- c/4 translation(d)
(  h  k  h)     k=2n : (-101) glide plane with b/2 translation (b,n)
(  h  k  h)  2h+k=4n : (-101) glide plane with +-a/4 +- b/4 + c/4 translation(d)
( -h  k  h)     k=2n : (101)  glide plane with b/2 translation (b,n)
( -h  k  h)  2h+k=4n : (101)  glide plane with +-a/4 +- b/4 + c/4 translation(d)
# ! monoclinic, ortho., tetra and cubic
(h 0 0)      h=2n : screw axis // [100] with  a/2 translation (21)
# ! cubic
(h 0 0)      h=2n : screw axis // [100] with 2a/4 translation (42)
# ! cubic
(h 0 0)      h=4n : screw axis // [100] with  a/4 translation (41)
# ! cubic
(h 0 0)      h=4n : screw axis // [100] with 3a/4 translation (43)
# ! monoclinic, ortho., tetra and cubic
(0 k 0)      k=2n : screw axis // [010] with  b/2 translation (21)
# ! cubic
(0 k 0)      k=2n : screw axis // [010] with 2b/4 translation (42)
# ! cubic
(0 k 0)      k=4n : screw axis // [010] with  b/4 translation (41)
# ! cubic
(0 k 0)      k=4n : screw axis // [010] with 3b/4 translation (43)
# ! monoclinic, ortho., tetra and cubic
(0 0 l)      l=2n : screw axis // [00l] with  c/2 translation (21)
# ! tetragonal and cubic
(0 0 l)      l=2n : screw axis // [00l] with 2c/4 translation (42)
# ! tetragonal and cubic
(0 0 l)      l=4n : screw axis // [00l] with  c/4 translation (41)
# ! tetragonal and cubic
(0 0 l)      l=4n : screw axis // [00l] with 3c/4 translation (43)
(0 0 0 l)    l=2n : screw axis // [00l] axis with 3c/6 translation (63)
(0 0 0 l)    l=3n : screw axis // [00l] axis with  c/3 translation (31)
(0 0 0 l)    l=3n : screw axis // [00l] axis with 2c/3 translation (32)
(0 0 0 l)    l=3n : screw axis // [00l] axis with 2c/6 translation (62)
(0 0 0 l)    l=3n : screw axis // [00l] axis with 4c/6 translation (64)
(0 0 0 l)    l=6n : screw axis // [00l] axis with  c/6 translation (61)
(0 0 0 l)    l=6n : screw axis // [00l] axis with 5c/6 translation (65)"""

Hkl_Ref_Conditions = Hkl_Ref_Conditions.split('\n')
Hkl_Ref_Conditions = [i for i in Hkl_Ref_Conditions if i[0] != '#']


def Search_Extinctions_Iunit(Spacegroup, Iunit):
    """!---- Arguments ----!
    type (Space_Group_Type), intent(in)     :: spacegroup
    integer,                 intent(in)     :: Iunit
    """

    Integral_Conditions(Spacegroup, Iunit)
    Glide_Planes_Conditions(Spacegroup, Iunit)
    Screw_Axis_Conditions(Spacegroup, Iunit)

    return


def is_equal(hkl1, hkl2):
    val = abs(np.array(hkl1) - np.array(hkl1))
    return True if np.all(val < eps_ref) else False


def hkl_absent(HKL, Spacegroup):
    for rot, trans in Spacegroup.get_symop():
        # multiply only by the rotational part
        opHKL = np.dot(HKL, rot)
        if is_equal(HKL, opHKL):
            r1 = np.dot(trans, HKL)
            if abs(r1 - round(r1)) > eps_ref:
                return True
    return False


def Integral_Conditions(SpaceGroup, iunit):
    """!---- Arguments ----!
    type (Space_Group_Type),  intent(in)     :: spacegroup
    integer, optional,        intent(in)     :: iunit

    # !---- local variables ----!
    integer               :: h, k,l, m
    integer               :: n, n_ext
    integer, dimension(3) :: hh
    integer               :: num_exti
    logical               :: integral_condition
    """

    integral_condition = False

    # 1.       h+k   = 2n                   C-face centred                      C
    # 2.       k+l   = 2n                   A-face centred                      A
    # 3.       h+l   = 2n                   B-face centred                      B
    # 4.       h+k+l = 2n                   Body centred                        I
    #
    # 5.       h+k   = 2n
    #      and k+l   = 2n
    #      and h+l   = 2n                   All-face centred                    F
    #     or h,k,l all odd
    #     or h,k,l all even
    #
    # 6.      -h+k+l = 3n                   Rhombohedrally centred,             R
    #                                       obverse setting

    if (iunit):
        print(" ")
        print(" >>> Integral reflections conditions for centred lattices:")
        print("----------------------------------------------------------")
        print(" ")

    def gen_hkl(x):
        g = np.mgrid[-x: x + 1, -x: x + 1, -x: x + 1]
        return np.vstack(list(map(np.ravel, g)))

    def test_absent(hkl, cond):
        # lines of hkl with h+k = 2n + 1      #reflections pouvant obeir a
        # la regle d"extinction
        p_hkl = hkl.T[cond]
        # reflecions obeissant a la regle
        absent_hkl = [True for hh in p_hkl if hkl_absent(hh, SpaceGroup)]
        return len(p_hkl) == len(absent_hkl)

    def end_action(num_exti):
        nonlocal integral_condition
        if iunit:
            print(f"#  {num_exti:3d} :  {Hkl_Ref_Conditions[num_exti]}")
        integral_condition = True

    # !---- C-face centred ----!
    # !  Hkl_Ref_Conditions(1) =   "(h k l)  h+k=2n           : xy0 centered base"
    num_exti = 1
    hkl = gen_hkl(6)  # colomn of hkl
    cond = np.asarray((hkl[0] + hkl[1]) % 2, dtype=bool)
    if test_absent(hkl, cond):
        end_action(num_exti)

    # !---- A-face centred ----!
    # !   Hkl_Ref_Conditions(2) =   "(h k l)  k+l=2n           : 0yz centered base"
    num_exti = 2
    hkl = gen_hkl(6)  # colomn of hkl
    cond = np.asarray((hkl[1] + hkl[2]) % 2, dtype=bool)
    if test_absent(hkl, cond):
        end_action(num_exti)

    # !---- B-face centred ----!
    # !  Hkl_Ref_Conditions(3) =   "(h k l)  h+l=2n           : x0z centered base"
    num_exti = 3
    hkl = gen_hkl(6)  # colomn of hkl
    cond = np.asarray((hkl[0] + hkl[2]) % 2, dtype=bool)
    if test_absent(hkl, cond):
        end_action(num_exti)

    # !---- Body centred (I) ----!
    # !  Hkl_Ref_Conditions(4) =   "(h k l)  h+k+l=2n         : body centred"
    num_exti = 4
    hkl = gen_hkl(3)  # colomn of hkl
    cond = np.asarray(hkl.sum(0) % 2, dtype=bool)
    if test_absent(hkl, cond):
        end_action(num_exti)
        return
    # !---- all-face centred (F) ----!
    # ! Hkl_Ref_Conditions(5) =   "(h k l)  h,k,l same parity: all-face cent"
    num_exti = 5
    hkl = gen_hkl(6)  # colomn of hkl
    c1 = np.sum(hkl % 2, axis=0)
    cond = np.asarray(np.logical_or((c1 == 1), (c1 == 2)))
    if test_absent(hkl, cond):
        end_action(num_exti)

    # !---- R network ----!
    # !  Hkl_Ref_Conditions(6) =   "(h k l) -h+k+l=3n         : Rhombohedrally centred (R)"
    num_exti = 6
    hkl = gen_hkl(6)  # colomn of hkl
    cond = np.asarray((-hkl[0] + hkl[1] + hkl[2]) % 3, dtype=bool)
    if test_absent(hkl, cond):
        end_action(num_exti)

    if not integral_condition:
        if iunit:
            print("     =====>>> no general reflection condition")
    return


def Screw_Axis_Conditions(SpaceGroup, Iunit):
    """
    # !---- Arguments ----!
    type (Space_Group_Type),       intent(in)     :: spacegroup
    integer, optional,             intent(in)     :: iunit

    # !---- Local variables ----!
    integer               :: h, k,l
    integer               :: n, n_ext
    integer, dimension(3) :: hh
    integer               :: num_exti
    logical               :: serial_condition
    """

    serial_condition = False

    def gen_hkl(x):
        g = np.mgrid[-x: x + 1, -x: x + 1, -x: x + 1]
        return np.vstack(map(np.ravel, g))

    def test_absent(hkl, cond):
        # lines of hkl with h+k = 2n + 1      #reflections pouvant obeir a
        # la regle d"extinction
        p_hkl = hkl.T[cond]
        # reflecions obeissant a la regle
        absent_hkl = [hh for hh in p_hkl if hkl_absent(hh, SpaceGroup)]
        return len(p_hkl) == len(absent_hkl)

    def end_action(num_exti):
        nonlocal serial_condition
        if Iunit:
            print(f"#  {num_exti:3d} :  {Hkl_Ref_Conditions[num_exti]}")
            serial_condition = True

    def screw_21_41(xyz, num_exti):
        """
          xyz = 0 , 1, 2
        """
        hkl = np.zeros((3, 13))
        hkl[xyz] = np.arange(-6, 7)
        cond = np.asarray(hkl[xyz] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

    def screw_42_43(xyz, num_exti):
        """
          xyz = 0 , 1, 2
        """
        hkl = np.zeros((3, 13))
        hkl[xyz] = np.arange(-6, 7)
        cond = np.asarray(hkl[xyz] % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)
            end_action(num_exti + 1)

    if (Iunit):
        print(" ")
        print(" >>> Serial reflections conditions for screw axes:")
        print("---------------------------------------------------")
        print(" ")

    # !SCREW AXES:      33 extinctions
    # check if "Monoclinic", "Orthorhombic" "Tetragonal" "Cubic"
    if (SpaceGroup.no in range(1, 143)) or (SpaceGroup.no > 194):
        # ! Hkl_Ref_Conditions(40) =   "(h 0 0)      h=2n : screw axis // [100] with  a/2 translation (21) # noqa E501
        num_exti = 40
        screw_21_41(0, num_exti)
        # ! Hkl_Ref_Conditions(44) =   "(0 k 0)      k=2n : screw axis // [010] with  b/2 translation (21) # noqa E501
        num_exti = 44
        screw_21_41(1, num_exti)
        # ! Hkl_Ref_Conditions(48) =   "(0 0 l)      l=2n : screw axis // [00l] with  c/2 translation (21) # noqa E501
        num_exti = 48
        screw_21_41(2, num_exti)

    if (SpaceGroup.no > 194):  # "Cubic" then
        # ! 41
        # ! Hkl_Ref_Conditions(41) =   "(h 0 0)      h=2n : screw axis // [100] with  2a/4 translation (42)" # noqa E501
        num_exti = 41
        screw_21_41(0, num_exti)

        # Hkl_Ref_Conditions(42) =   "(h 0 0)      h=4n : screw axis // [100] with  a/4 translation (41)" # noqa E501
        # Hkl_Ref_Conditions(43) =   "(h 0 0)      h=4n : screw axis //
        # [100] with 3a/4 translation (43)"   # ! cubic
        num_exti = 42
        screw_42_43(0, num_exti)

        # ! Hkl_Ref_Conditions(45) =   "(0 k 0)      k=2n : screw axis // [010] with  2b/4 translation (42)"   # ! cubic
        num_exti = 45
        screw_21_41(1, num_exti)

        # ! Hkl_Ref_Conditions(46) =   "(0 k 0)      k=4n : screw axis // [010] with  b/4 translation (41)"   # ! cubic
        # ! Hkl_Ref_Conditions(47) =   "(0 k 0)      k=4n : screw axis // [010] with 3b/4 translation (43)"   # ! cubic
        num_exti = 46
        screw_42_43(1, num_exti)

        # ! Hkl_Ref_Conditions(49) =   "(0 0 l)      l=2n : screw axis // [00l] with  c/2 translation (21)"   # ! monoclinic, ortho. and cubic
        num_exti = 49
        screw_21_41(2, num_exti)

        # ! Hkl_Ref_Conditions(50) =  "(0 0 l)      l=4n : screw axis // [00l] with  c/4 translation (41)"   # ! tetragonal and cubic
        # ! Hkl_Ref_Conditions(51) =  "(0 0 l)      l=4n : screw axis // [00l] with 3c/4 translation (43)"   # ! tetragonal and cubic
        num_exti = 50
        screw_42_43(2, num_exti)

    if (SpaceGroup.no in range(143, 195)):   # hexagonal cell --> Rombo + Hexagonal groups
        # ! Hkl_Ref_Conditions(52) =   "(0 0 0 l)    l=2n : screw axis // [00l] axis with 3c/6 translation (63)"
        num_exti = 52
        screw_21_41(2, num_exti)

        # ! Hkl_Ref_Conditions(53) =   "(0 0 0 l)    l=3n : screw axis // [00l] axis with  c/3 translation (31)"
        # ! Hkl_Ref_Conditions(54) =   "(0 0 0 l)    l=3n : screw axis // [00l] axis with 2c/3 translation (32)"
        # ! Hkl_Ref_Conditions(55) =   "(0 0 0 l)    l=3n : screw axis // [00l] axis with 2c/6 translation (62)"
        # ! Hkl_Ref_Conditions(56) =   "(0 0 0 l)    l=3n : screw axis // [00l] axis with 4c/6 translation (64)"
        num_exti = 53
        hkl = np.zeros((3, 13))
        hkl[2] = np.arange(-6, 7)
        cond = np.asarray(hkl[2] % 3, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)
            end_action(num_exti + 1)
            if (SpaceGroup.no in range(168, 195)):
                end_action(num_exti + 2)
                end_action(num_exti + 3)

    if (SpaceGroup.no in range(168, 195)):   # Hexagonal groups
        # ! Hkl_Ref_Conditions(57)="(0 0 0 l)    l=6n : screw axis // [00l] axis with  c/6 translation (61)"
        # ! Hkl_Ref_Conditions(58)="(0 0 0 l)    l=6n : screw axis // [00l] axis with 5c/6 translation (65)"
        num_exti = 57
        hkl = np.zeros((3, 13))
        hkl[2] = np.arange(-6, 7)
        cond = np.asarray(hkl[2] % 6, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)
            end_action(num_exti + 1)

    if not serial_condition:
        if Iunit:
            print("     =====>>> no serial reflection condition")


def Glide_Planes_Conditions(SpaceGroup, Iunit):
    """
    # !---- Arguments ----!
    type (Space_Group_Type), intent(in)     :: spacegroup
    integer, optional,       intent(in)     :: iunit

    # !---- Local variables ----!
    integer               :: h, k,l, m
    integer               :: n, n_ext
    integer, dimension(3) :: hh
    integer               :: num_exti
    logical               :: zonal_condition
    """
    zonal_condition = False

    def gen_hkl(x):
        g = np.mgrid[-x: x + 1, -x: x + 1, -x: x + 1]
        return np.vstack(map(np.ravel, g))

    def gen_hk0(zero, x):
        g = np.mgrid[-x: x + 1, -x: x + 1]
        hkl = np.vstack(list(map(np.ravel, g)))
        hkl = np.insert(hkl, zero, np.zeros_like(hkl[0]), 0)
        return hkl

    def gen_hhl(l_diff, x):
        g1, g2 = np.mgrid[-x: x + 1, -x: x + 1]
        hkl = np.tile(g1, (3, 1))
        hkl[l_diff] = g2
        return hkl

    def test_absent(hkl, cond):
        p_hkl = hkl.T[cond]   # la regle d"extinction
        absent_hkl = [hh for hh in p_hkl if hkl_absent(
            hh, SpaceGroup)]  # ref. good with rule
        return len(p_hkl) == len(absent_hkl)

    def end_action(num_exti):
        nonlocal zonal_condition
        if Iunit:
            print(f"#  {num_exti:3d} :  {Hkl_Ref_Conditions[num_exti]}")
        zonal_condition = True

    def glide_abc(tras, mir, num_exti):
        """
          xyz = 0 , 1, 2
        """
        g = np.mgrid[-6:7, -6:7]
        hkl = np.vstack(list(map(np.ravel, g)))
        hkl = np.insert(hkl, mir, np.zeros_like(hkl[0]), 0)
        cond = np.asarray(hkl[tras] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

    if Iunit:
        print(" ")
        print(" >>> Zonal reflections conditions for glide planes:")
        print("---------------------------------------------------")
        print(" ")

    # !GLIDE PLANES and screw axes: table 2.13.2
        # !-------------
        # !
        # !        0 k l:    k=2n    b/2             monoclinic, orthorhombic, tetragonal and cubic
        # !        0 k l:    l=2n    c/2             monoclinic, orthorhombic, tetragonal and cubic
        # !        0 k l:  k+l=2n    b/2 +  c/2      monoclinic, orthorhombic, tetragonal and cubic
        # !        0 k l:  k+l=4n    b/4 +- c/4      orthorhombic and cubic
        # !
        # !
        # !        h 0 l:    h=2n    a/2             monoclinic, orthorhombic, tetragonal and cubic
        # !        h 0 l:    l=2n    c/2             monoclinic, orthorhombic, tetragonal and cubic
        # !        h 0 l:  l+h=2n    c/2 +  a/2      monoclinic, orthorhombic, tetragonal and cubic
        # !        h 0 l:  l+h=4n    c/4 +- a/4      orthorhombic and cubic
        # !
        # !        h k 0:    h=2n    a/2             monoclinic, orthorhombic, tetragonal and cubic
        # !        h k 0:    k=2n    b/2             monoclinic, orthorhombic, tetragonal and cubic
        # !        h k 0:  h+k=2n    a/2 +  b/2      monoclinic, orthorhombic, tetragonal and cubic
        # !        h k 0:  h+k=4n    a/4 +- b/4      monoclinic, orthorhombic, tetragonal and cubic

    # check if "Monoclinic", "Orthorhombic" "Tetragonal" "Cubic"
    if (SpaceGroup.no in range(1, 143)) or (SpaceGroup.no > 194):

        # !---- glide plane b/2:
        # ! Hkl_Ref_Conditions(7)  =   "(0 k l)      k=2n : 0yz glide plane with b/2 translation"
        num_exti = 7
        glide_abc(tras=1, mir=0, num_exti=num_exti)

        # !---- glide plane c/2:
        # ! Hkl_Ref_Conditions(8)  =   "(0 k l)      l=2n : 0yz glide plane with c/2 translation"
        num_exti = 8
        glide_abc(tras=2, mir=0, num_exti=num_exti)

        # !---- glide plane b/2 + c/2:
        # !Hkl_Ref_Conditions(9)  =   "(0 k l)    k+l=2n : 0yz glide plane with b/2 + c/2 translation"
        num_exti = 9
        hkl = gen_hk0(0, 6)  # colomn of hkl
        cond = np.asarray((hkl[1] + hkl[2]) % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane a/2:
        # !  Hkl_Ref_Conditions(11)  =   "(h 0 l)      h=2n : x0z glide plane with a/2 translation"
        num_exti = 11
        glide_abc(tras=0, mir=1, num_exti=num_exti)

        # !---- glide plane c/2:
        # ! Hkl_Ref_Conditions(12) =   "(h 0 l)      l=2n : x0z glide plane with c/2 translation"
        num_exti = 12
        glide_abc(tras=2, mir=1, num_exti=num_exti)

        # !---- glide plane c/2 + a/2:
        # ! Hkl_Ref_Conditions(13) =   "(h 0 l)    l+h=2n : x0z glide plane with a/2 + c/2 translations"
        num_exti = 13
        hkl = gen_hk0(1, 6)  # colomn of hkl
        cond = np.asarray((hkl[0] + hkl[2]) % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane a/2:
        # ! Hkl_Ref_Conditions(15) =   "(h k 0)      h=2n : xy0 glide plane with a/2 translation"
        num_exti = 15
        glide_abc(tras=0, mir=2, num_exti=num_exti)

        # !---- glide plane b/2:
        # !Hkl_Ref_Conditions(16) =   "(h k 0)      k=2n : xy0 glide plane with b/2 translation"
        num_exti = 16
        glide_abc(tras=1, mir=2, num_exti=num_exti)

        # !---- glide plane a/2 + b/2:
        # ! Hkl_Ref_Conditions(17) =   "(h k 0)    h+k=2n : xy0 glide plane with a/2 + b/2 translations"
        num_exti = 17
        hkl = gen_hk0(2, 6)  # colomn of hkl
        cond = np.asarray((hkl[0] + hkl[1]) % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

    # check if  "Orthorhombic" .or. "Cubic"
    if (SpaceGroup.no in range(16, 75)) or (SpaceGroup.no > 194):

        # !---- glide plane b/4 + c/4:
        # ! Hkl_Ref_Conditions(10)  =   "(0 k l)    k+l=4n : 0yz glide plane with b/4 +- c/4 translation"
        num_exti = 10
        hkl = gen_hk0(0, 6)  # colomn of hkl
        cond = np.asarray((hkl[1] + hkl[2]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane c/4 + a/4:
        # ! Hkl_Ref_Conditions(14) =   "(h 0 l)    l+h=4n : x0z glide plane with a/4 +- c/4 translations"
        num_exti = 14
        hkl = gen_hk0(1, 6)  # colomn of hkl
        cond = np.asarray((hkl[0] + hkl[2]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane a/4 + b/4:
        # ! Hkl_Ref_Conditions(18) =   "(h k 0)    h+k=4n : xy0 glide plane with a/4 +- b/4 translations"
        num_exti = 18
        hkl = gen_hk0(2, 6)  # colomn of hkl
        cond = np.asarray((hkl[0] + hkl[1]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)
    # end if  ! fin de la condition "if ortho, cubic

    if (SpaceGroup.no in range(143, 195)):   # hexagonal cell --> Rombo + Hexagonal groups
        # !---- glide plane with c/2 translation: hexagonal
        # !  Hkl_Ref_Conditions(19) =   "(  h  -h   0 l) l=2n : (11-20) glide plane with c/2 translation (c)"
        num_exti = 19
        hkl = gen_hhl(2, 6)
        hkl[1] *= -1
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with c/2 translation: hexagonal
        # !  Hkl_Ref_Conditions(20) =   "(  0   k  -k l) l=2n : (-2110) glide plane with c/2 translation (c)"
        num_exti = 20
        hkl = gen_hk0(0, 6)
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

       # !---- glide plane with c/2 translation: hexagonal
       # !Hkl_Ref_Conditions(21) =   "( -h   0   h l) l=2n : (1-210) glide plane with c/2 translation (c)"
        num_exti = 21
        hkl = gen_hk0(1, 6)
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with c/2 translation: hexagonal
        # ! Hkl_Ref_Conditions(22) =   "(  h   h -2h l) l=2n : (1-100) glide plane with c/2 translation (c)"
        num_exti = 22
        hkl = gen_hhl(2, 6)
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with c/2 translation: hexagonal
        # !  Hkl_Ref_Conditions(23) =   "(-2h   h   h l) l=2n : (01-10) glide plane with c/2 translation (c)"
        num_exti = 23
        hkl = gen_hhl(2, 6)
        hkl[0] *= -2
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with c/2 translation: hexagonal
        # !  Hkl_Ref_Conditions(24) =   "(  h -2h   h l) l=2n : (-1010) glide plane with c/2 translation (c)"
        num_exti = 24
        hkl = gen_hhl(2, 6)
        hkl[1] *= -2
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !25: glide plane with c/2 translation: rhomboedral
        # !  Hkl_Ref_Conditions(25) =  "(  h  h  l) l=2n : (1-10) glide plane with c/2 translation (c,n)"
        num_exti = 25
        hkl = gen_hhl(2, 6)
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with c/2 translation: rhomboedral
        # !  Hkl_Ref_Conditions(26) =  "(  h  k  k) h=2n : (01-1) glide plane with a/2 translation (a,n)"
        num_exti = 26
        hkl = gen_hhl(0, 6)
        cond = np.asarray(hkl[0] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !27: glide plane with c/2 translation: rhomboedral
        # !  Hkl_Ref_Conditions(27) =  "(  h  k  h) k=2n : (-101) glide plane with b/2 translation (b,n)"
        num_exti = 27
        hkl = gen_hhl(1, 6)
        cond = np.asarray(hkl[1] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

    # check if  "Tetragonal" .or.  "Cubic"
    if (SpaceGroup.no in range(75, 143)) or (SpaceGroup.no > 195):
        # !---- glide plane with c/2 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(28) =  "(  h  h  l)    l=2n : (1-10) glide plane with c/2 translation (c,n)"
        num_exti = 28
        hkl = gen_hhl(2, 6)
        cond = np.asarray(hkl[2] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with a/4 +- b/4 +- c/4 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(29) =  "(  h  h  l) 2h+l=4n : (1-10) glide plane with a/4 +- b/4 +- c/4 translation (d)"
        num_exti = 29
        hkl = gen_hhl(2, 6)
        cond = np.asarray((2 * hkl[0] + hkl[2]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !30: glide plane with c/2 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(30) =  "(  h -h  l)    l=2n : (110)  glide plane with c/2 translation (c,n)"
        num_exti = 30
        hkl = gen_hhl(2, 6)
        hkl[1] *= -1
        cond = np.asarray(hkl[2] % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # ! 31: glide plane with a/4 +- b/4 +- c/4 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(31) = "(  h -h  l) 2h+l=4n : (110)  glide plane with a/4 +- b/4 +- c/4 translation (d)"
        num_exti = 31
        hkl = gen_hhl(2, 6)
        hkl[1] *= -1
        cond = np.asarray((2 * hkl[0] + hkl[2]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

    # check if  "Cubic"
    if (SpaceGroup.no > 195):
        # !---- glide plane with a/2 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(32) = "(  h  k  k)    h=2n : (01-1) glide plane with a/2 translation (a,n)"
        num_exti = 32
        hkl = gen_hhl(0, 6)
        cond = np.asarray(hkl[0] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !---- glide plane with +-a/4 +- b/4 +- c/4 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(33) = "(  h  k  k) 2k+h=4n : (01-1) glide plane with +-a/4 + b/4 +- c/4 translation (d)"
        num_exti = 33
        hkl = gen_hhl(0, 6)
        cond = np.asarray((2 * hkl[1] + hkl[0]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !34: glide plane with a/2 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(34) =  "(  h  k -k)    h=2n : (011)  glide plane with a/2 translation (a,n)"
        num_exti = 34
        hkl = gen_hhl(0, 6)
        hkl[2] *= -1
        cond = np.asarray(hkl[0] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # ! 35: glide plane with a/4 +- b/4 +- c/4 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(351) = "(  h  k -k) 2k+h=4n : (011)  glide plane with +-a/4 + b/4 +- c/4 translation (d)"
        num_exti = 35
        hkl = gen_hhl(0, 6)
        hkl[2] *= -1
        cond = np.asarray((2 * hkl[1] + hkl[0]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !36: glide plane with b/2 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(36) = "(  h  k  h)    k=2n : (-101) glide plane with b/2 translation (b,n)"
        num_exti = 36
        hkl = gen_hhl(1, 6)
        cond = np.asarray(hkl[1] % 2, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !37: glide plane with +-a/4 +- b/4 +- c/4 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(33) = "(  h  k  h) 2h+k=4n : (-101) glide plane with +-a/4 + b/4 +- c/4 translation (d)"
        num_exti = 37
        hkl = gen_hhl(1, 6)
        cond = np.asarray((2 * hkl[0] + hkl[1]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # !38: glide plane with b/2 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(38) = "( -h  k  h)    k=2n : (101)  glide plane with b/2 translation (b,n)"
        num_exti = 38
        hkl = gen_hhl(1, 6)
        hkl[0] *= -1
        cond = np.asarray(hkl[1] % 2, dtype=bool)

        # ! 39: glide plane with a/4 +- b/4 +- c/4 translation: tetragonal + cubic
        # !  Hkl_Ref_Conditions(39) = "( -h  k  h) 2h+k=4n : (101)  glide plane with +-a/4 + b/4 +- c/4 translation (d)"
        num_exti = 39
        hkl = gen_hhl(1, 6)
        hkl[0] *= -1
        cond = np.asarray((2 * hkl[0] + hkl[1]) % 4, dtype=bool)
        if test_absent(hkl, cond):
            end_action(num_exti)

        # end if  ! fin de la condition "if cubic

    if not zonal_condition:
        if Iunit:
            print("     =====>>> no zonal reflection condition")
