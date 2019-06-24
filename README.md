<!--https://www.codecogs.com/latex/eqneditor.php-->


#Project name: AI-LJ

## Creating data set for training

## inBasket function
inBasket is a function that determines whether or not a trajectory intersects a basket. The determination is analytical, and correct.
The function receives the initial speeds (in the x and y direction), the gravitational constant (g), and a basket. The basket is defined as two points in space, so that the line segment between them is the basket itself. If the trajectory intersects this line segment, the function returns a positive answer. If the trajectory does not intersect with the basket, it returns a negative answer.
To determine the intersection, the function does the following:
1)	The trajectory parabola is found by substituting for time
2)	The infinite basket line is found by two points
3)	The line and parabola are compared to find coincident points
4)	Coincident points are checked. If one is in the basket line segment, a true answer is returned

The trajectory parabola was originally represented by x, y and time:
![x=v0*t](https://latex.codecogs.com/gif.latex?x%3DV_%7B0%7D%5Ccdot%20t)
$y=v_{0,y}t + \frac{1}{2}gt^2$
Substituting t to solve for x, y at any time:
$t=\frac{x}{v_{x,0}}$
$y=\frac{v_{y,0}}{v_{x,0}}x + \frac{g}{2v_{x,0}^2}x^2$
Simplifying:
a=\frac{g}{2v_{x,0}^2},\\ b=\frac{v_{y,0}}{v_{x,0}},\\ c=0$
We have a simple parabolic equation:
$y_{parab}=ax^2+bx+c$
Next we find the line equation by two points- $(x_{b1},y_{b1}), (x_{b2},y_{b2})$. The slope of the line is:
$m=\frac{y_{b2}-y_{b1}}{x_{b2}-x_{b1}}$
The y intercept is:
$n=-mx_1+y_1$
Giving us the line:
$y_{line}=mx+n$
To find intersecting points, we will compare $y_{parab}=y_{line}$, and solve for 0:
$0= ax^2+bx+c –mx –n$
Let’s rename the coefficients of x:
$A = a,\\ B =b-m,\\ C=c-n$
$Ax^2+Bx+C=0$
We can find the 0-intercept points by solving the quadratic equation:
$x_{1,2}=\frac{-B\pm\sqrt{B^2-4AC}}{2A}$
First we test the discriminant to find if there are intersection points at all. If $B^2-4AC<0$, there are no interception points and the function returns a false answer. The quadratic equation is not applied, because it would result in an imaginary number that has no value to us. If the discriminant is greater or equal to zero, there is one or two intersection points. We find their x coordinated by solving the quadratic equation.
We find the corresponding y values of these points by substituting $x_{1,2}$ in the line or parabolic equation.
$y_1=ax_1^2+bx_1+c$
$y_2=ax_2^2+bx_2+c$
We then check if $(x_1,y_1)$ or $(x_2,y_2)$ are within the interval $(x_{b1},y_{b1}), (x_{b2},y_{b2})$. This is done by component, i.e., $x_1 > min(x_{b1},x_{b2})$.
If either of the points does intersect the basket, “True” is returned. Otherwise- “False”.

## Testing - Results

## Conclusions

<img src="table1.png" align="center" width=300>


![a*x^2+b*x+c=0](https://latex.codecogs.com/gif.latex?A%5Ccdot%20x%5E2&plus;B%5Ccdot%20x&plus;C%3D0)

![0=A\cdotx^2+B\cdot x+C-(m\cdot x+n)](https://latex.codecogs.com/gif.latex?0%3DA%5Ccdot%20x%5E2&plus;B%5Ccdot%20x&plus;C-%28m%5Ccdot%20x&plus;n%29)

