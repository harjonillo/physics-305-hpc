! to compile this into something python can understand, run this in the terminal:
! f2py -c hotdog.f90 -m hotdog

subroutine bacon(ed, philip, rene)
    real(8), intent(in) :: ed, philip  ! variable type/intent needs to be declared. options: input, output, dummy
    real(8), intent(out) :: rene  !
    rene = ed + philip
end subroutine

subroutine waffle(x, a, N)
    implicit none
    real(8), intent(in) :: x(N)
    real(8), intent(out) :: a(N)
    integer(8), intent(in) :: N
    integer(8) :: i, j
    a = 0.0
    do i = 1, N
        do j = 1, i-1
            a(i) = a(i) + (x(i) - x(j))**(-2)
        end do
        do j = i+1, N
            a(i) = a(i) + (x(i) - x(j))**(-2)
        end do
    end do
end subroutine
