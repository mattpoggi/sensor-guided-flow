#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float gaussian(float x, float y, float v, float k, float c)
{
	return 1-v + v*k*exp(-(x*x+y*y)/(2*c*c));
}

//unsigned char *forward_warping(const void *src, const void *idx, const void *idy, const void *z, int h, int w)
void build_guide(const void *flow_x, const void *flow_y, const void *flow_valid, void *result, const int b, const int h, const int w, const float k, const float c)
{
	float *forward_x = (float *)calloc(b * h * w, sizeof(float));
	float *forward_y = (float *)calloc(b * h * w, sizeof(float));
	for (int z = 0; z < b; z++)
                #pragma omp parallel for collapse(2)
		for (int y0 = 0; y0 < h; y0++)
			for (int x0 = 0; x0 < w; x0++)
			{
				if ( ((float*)flow_valid)[z*w*h + y0*w + x0] != 0)
				{
					forward_x[z*w*h + y0*w + x0] = ((float*)flow_x)[z*w*h + y0*w + x0] + x0;
					forward_y[z*w*h + y0*w + x0] = ((float*)flow_y)[z*w*h + y0*w + x0] + y0;
				}
				for (int y1 = 0; y1 < h; y1++)
		        	        for (int x1 = 0; x1 < w; x1++)
					{
						// Accessing cell y=y0*w+x0 , x=y1*w+x1
						// y term multiplied by row length (h*w)
						((float*)result)[z*w*h*w*h + (y0*w+x0)*w*h + y1*w+x1] = gaussian(x1-forward_x[z*w*h + y0*w + x0], y1-forward_y[z*w*h + y0*w + x0], ((float*)flow_valid)[z*w*h + y0*w + x0], k, c);
					}
			}
	free(forward_x);
	free(forward_y);
	return;
}
