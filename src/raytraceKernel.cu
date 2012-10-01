// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include "EasyBMP.h"
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hashF(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	  ray r;
	  r.origin = eye;
	  glm::vec3 aVector = glm::cross(view, up);
	  glm::vec3 bVector = glm::cross(aVector, view);
	  glm::vec3 midPointVector = eye + view;
	  glm::vec3 horizontalVector = (aVector * (float)(view.length() * tan(fov.x ))) / (float) aVector.length();
	  glm::vec3 verticalVector = (bVector * (float)(view.length() * tan(fov.y ))) / (float) bVector.length();
	  float tempX = (float)x / (float) (resolution.x -1 );
	  float tempY = (float)y / (float) (resolution.y -1);
	  glm::vec3 pointVector = midPointVector + ((float)(2.0f*tempX -1) * horizontalVector) + ((float)(2.0 * tempY - 1) * verticalVector);
	  r.direction = glm::normalize(pointVector - eye);
	


	  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__host__ __device__ float calculateLights(staticGeom* geoms,int numberOfGeoms, staticGeom light,glm::vec3 intersectionPoint, float lightType, int numberOfLights, float &dist, int indexLight){
	  float avg = 0; 
	  glm::vec3 tempNormal = glm::vec3(0);
	  float hitPointDistance = -1;
	  glm::vec3 tempIntersectionPoint = glm::vec3(0);
	  float closestIntersectionDistance = 100000.0f;
	  if(lightType>0){
		 
				//check intersection with geometry
				for ( int p = 0; p < 9; p++ )// for ( int q = 0; q < 3; q++ )
				{
						ray tempLightRay;
						
						tempLightRay.direction = glm::normalize(getRandomPointOnCube(light,p) - intersectionPoint);
						tempLightRay.origin = intersectionPoint;
																		
						tempNormal = glm::vec3(0);
						hitPointDistance = -1;
						tempIntersectionPoint = glm::vec3(0);
						for(int i = 0; i < numberOfGeoms; i++)
						{
																		
							if(light.translation != geoms[i].translation && i!= indexLight ){
									if( geoms[i].type == SPHERE )
									{
										hitPointDistance = sphereIntersectionTest(geoms[i], tempLightRay, tempIntersectionPoint, tempNormal);

									}else if(geoms[i].type == CUBE)
									{
										hitPointDistance = boxIntersectionTest(geoms[i], tempLightRay, tempIntersectionPoint, tempNormal);
					      
									}
									if( hitPointDistance != -1 && hitPointDistance < closestIntersectionDistance )
									{
										dist = hitPointDistance;
										break;
									}
																					
				
												
								}
						}//end geometry for loop
						if( hitPointDistance == -1  )
						{
							//lightDist = hitPointDistance;
							//break;
							avg+=(1.0/9.0);
						}
				}
		}else{
			ray lightRay;
			
			lightRay.direction = glm::normalize(light.translation - intersectionPoint);
			lightRay.origin = intersectionPoint;

			{
					tempNormal = glm::vec3(0);
					hitPointDistance = -1;
					tempIntersectionPoint = glm::vec3(0);
					avg = 1;
					for(int i = 0; i < numberOfGeoms; i++)
					{
						if(light.translation != geoms[i].translation && i!= indexLight ){
								if( geoms[i].type == SPHERE )
								{
									hitPointDistance = sphereIntersectionTest(geoms[i], lightRay, tempIntersectionPoint, tempNormal);

								}else if(geoms[i].type == CUBE)
								{
									hitPointDistance = boxIntersectionTest(geoms[i], lightRay, tempIntersectionPoint, tempNormal);
					      
								}
								if( hitPointDistance != -1 && hitPointDistance < closestIntersectionDistance )
								{
									dist = hitPointDistance;
									avg = 0;
									break;
								}
				
												
							}
					}//end geometry for loop
			}


		}//point light end


	  return avg;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors1, 
                            staticGeom* geoms, int numberOfGeoms, material* cudaMat, int numberOfMaterials,
							int* cudaLights, int numberOfLights, float * cudaRed, float * cudaGreen, float * cudaBlue, int width, int height ){

	    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int index = x + (y * resolution.x);
		

		if((x<=resolution.x && y<=resolution.y)){
		 colors1[index]= glm::vec3(0);
		 for (float fx = float(x) ; fx < x + 1.0f; fx += 0.5f )
         for (float fy = float(y) ; fy < y + 1.0f; fy += 0.5f ){ 
			ray r = raycastFromCameraKernel(resolution, time, fx, fy, cam.position, cam.view, cam.up, cam.fov);
			float n1 = 1.0; //index of refraction of original medium
			float n2 = 1.0;//index of refraction of other medium
			float n = n1/n2; // ratio of index of refraction
			glm::vec3 textureCol = glm::vec3(0);
			//float lightDist = 100000.0f;
			float closestIntersectionDistance = 100000.0;
			float hitPointDistance = -1;
			int currentIndex = -1;
			glm::vec3 normal = glm::vec3(0);
			glm::vec3 tempNormal = glm::vec3(0);
			glm::vec3 intersectionPoint = glm::vec3(0);	
			glm::vec3 tempIntersectionPoint = glm::vec3(0);
			glm::vec3 transparency = glm::vec3(0,0,0);
					     
			glm::vec3 refrCol = glm::vec3(0);//refracted color
			glm::vec3 colors = glm::vec3( 0);// * currentMat.color;
			float component = 0.8f;
			int depth = 0; 
			do{
						intersectionPoint = glm::vec3(0);
						tempIntersectionPoint = glm::vec3(0);
						normal = glm::vec3(0);
						tempNormal = glm::vec3(0);
						closestIntersectionDistance = 100000.0;
						hitPointDistance = -1;
						currentIndex = -1;
						for(int i = 0; i < numberOfGeoms; i++)
						{
							if( geoms[i].type == SPHERE )
							{
								hitPointDistance = sphereIntersectionTest(geoms[i], r, tempIntersectionPoint, tempNormal);

							}else if(geoms[i].type == CUBE)
							{
								hitPointDistance = boxIntersectionTest(geoms[i], r, tempIntersectionPoint, tempNormal);
					      
							}
							//find the closest intersetion point
							if(hitPointDistance > 0.0)
							{
								if(hitPointDistance < closestIntersectionDistance)
								{
									closestIntersectionDistance = hitPointDistance;
									normal = tempNormal;
									intersectionPoint = tempIntersectionPoint;
									currentIndex = i;
								}
							}
						}
						
						if( currentIndex == -1 ){
		    				colors = glm::vec3(0);
							component = 0;
						}else{
				
										material currentMat = cudaMat[geoms[currentIndex].materialid];
									
										if(currentMat.texture >0){
														glm::vec3 Vn = glm::vec3(0,1,0); 
														glm::vec3 Ve = glm::vec3(1,0,0);
														
														//unit length vector vp
														glm::vec3 Vp=glm::normalize(intersectionPoint - geoms[currentIndex].translation);
													
														float phi = acos((  glm::dot(Vn,Vp))/(sqrtf(pow(Vn.x,2)+pow(Vn.y,2)+pow(Vn.z,2))*sqrtf(pow(Vp.x,2)+pow(Vp.y,2)+pow(Vp.z,2))));
														float v = phi / PI;
														float  theta = ( glm::dot(Vn,Ve)) /(sqrtf(pow(Ve.x,2)+pow(Ve.y,2)+pow(Ve.z,2))*sqrtf(pow(Vn.x,2)+pow(Vn.y,2)+pow(Vn.z,2)));/// sin( phi )) ) / ( 2 * PI);
														float theta2=acos(theta/ (sin( phi )))  / ( 2 * PI);
					
														glm::vec3 crossP = glm::cross(Vn,Ve);
														
														float u;
														if ( glm::dot(Vp,crossP) > 0 )
															u = theta2;
														else
															u = 1 - theta2;
											         

														int tempIndex = ((u * width)-1) * height+ ((v*height)-1);
														textureCol = glm::vec3(cudaRed[tempIndex], cudaGreen[tempIndex],cudaBlue[tempIndex]);
										}else
										    textureCol = currentMat.color;
										
										colors += glm::vec3( 0.1 ) * textureCol + refrCol * transparency;
							
              
										//if the emittance is greater than 0 then object is a light, hence it will have the color of its own
										if( currentMat.emittance > 0 )
										{
											colors = currentMat.color;
										}else{

												ray lightRay;
												//parse over lights
												for( int j = 0; j < numberOfLights; j++ )
												{
													float avg = calculateLights( geoms,numberOfGeoms,geoms[cudaLights[j]],intersectionPoint,cudaMat[geoms[cudaLights[j]].materialid].areaLight, numberOfLights,closestIntersectionDistance, cudaLights[j]);
													 
													   if(cudaMat[geoms[cudaLights[j]].materialid].areaLight>0){
														
																glm::vec3 currentLightPos = geoms[cudaLights[j]].translation + 0.5f * geoms[cudaLights[j]].scale ;// geoms[cudaLights[j]].translation;
																lightRay.direction = glm::normalize(currentLightPos - intersectionPoint);
																lightRay.origin = intersectionPoint;
																
													   }else{
														 
															lightRay.direction = glm::normalize(geoms[cudaLights[j]].translation - intersectionPoint);
															lightRay.origin = intersectionPoint;

													   }//point light end
												

														//not in shadow
														if (avg >= 0)
														{	
															//phong highlight
															float cosAngle = glm::max(glm::dot( r.direction, lightRay.direction - 2.0f * glm::dot( lightRay.direction, normal ) * normal ),0.0f);
															if (cosAngle > 0)
															{
																float spec = powf( cosAngle, currentMat.specularExponent) * glm::dot(currentMat.specularColor,textureCol) ;
																colors += spec * cudaMat[cudaLights[j]].color* avg;
															}
															float diffuse =(glm::max( (float) glm::dot( lightRay.direction, normal ),0.0f)) * component;
															colors += textureCol * avg*cudaMat[cudaLights[j]].color * diffuse ;
														}
												}//end light for loop

										}//not a light condition
										
										//reflection calculations
										if(currentMat.hasReflective > 0){
											component *= currentMat.hasReflective;
											float reflect = 2.0f * glm::dot(r.direction, normal);
											r.origin = intersectionPoint;
											r.direction = r.direction - reflect * normal;
										}else if ((currentMat.hasRefractive > 0))//refraction
										{
											n2 = currentMat.indexOfRefraction;
											n = n1 / n2;
											float cosAngle = glm::dot( normal, r.direction);
											float secondTerm = 1.0f - n * n * (1.0f - cosAngle * cosAngle);
											if (secondTerm >= 0.0f)
											{
												r.direction = (n * r.direction) - (n * cosAngle + sqrtf( secondTerm )) * normal;
												r.direction  = glm::normalize(r.direction );
												r.origin = intersectionPoint;
												component *= currentMat.hasRefractive;
												n1 = n2;
												glm::vec3 absorbance = currentMat.absorptionCoefficient* (- closestIntersectionDistance);
												transparency = glm::vec3( expf( absorbance.x ), expf( absorbance.y ), expf( absorbance.z) );
												refrCol += colors;
												
											}
										}else{
											component = 0;

										}
								}

					
						depth++; 
				}while((component > 0.0f) && (depth < 15));
					colors1[index] += colors/4.0f;
	            
			}	
		}//end x, y condition
		__syncthreads();
		
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);


  //lights
  int numberOfLights = 0;
  int* lightList = new int[numberOfGeoms];//number of lights can be equal to number of objects
  int numberOfTexture = 0;
  int *textureIDList = new int[numberOfGeoms];
  for(int i =0;i<numberOfGeoms;i++){
	  textureIDList[i]= -1;

  }
 // BMP *Input=new BMP[numberOfGeoms];
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
	//if emittance for the material assigned to object > 0 then it is a light source
	if( materials[newStaticGeom.materialid].emittance > 0 )
	{
		lightList[numberOfLights] = i;
		numberOfLights ++;
	}
	
  }
  
  BMP Input;
 // Input.ReadFromFile("renders/brick.bmp");
  float *redInput = new float[Input.TellWidth()*Input.TellHeight()];
  float *greenInput = new float[Input.TellWidth()*Input.TellHeight()];
  float *blueInput = new float[Input.TellWidth()*Input.TellHeight()];
  
 
/*  for(int i =0;i<Input.TellWidth() ;i++){
	  for(int j =0;j < Input.TellHeight() ;j++){
		  int index = i * Input.TellHeight() + j;
		
		  redInput[index] = Input(i,j)->Red/255.0;
		  greenInput[index] = Input(i,j)->Green/255.0;
		  blueInput[index] = Input(i,j)->Blue/255.0;
		


	  }

  }*/

 

  //package redColor
  float* cudaRed = NULL;
  cudaMalloc((void**)&cudaRed, Input.TellWidth()*Input.TellHeight()*sizeof(float));
  cudaMemcpy( cudaRed, redInput, Input.TellWidth()*Input.TellHeight()*sizeof(float), cudaMemcpyHostToDevice);

  //package blueColor
  float* cudaBlue = NULL;
  cudaMalloc((void**)&cudaBlue, Input.TellWidth()*Input.TellHeight()*sizeof(float));
  cudaMemcpy( cudaBlue, blueInput, Input.TellWidth()*Input.TellHeight()*sizeof(float), cudaMemcpyHostToDevice);

  //package greenColor
  float* cudaGreen = NULL;
  cudaMalloc((void**)&cudaGreen, Input.TellWidth()*Input.TellHeight()*sizeof(float));
  cudaMemcpy( cudaGreen, greenInput, Input.TellWidth()*Input.TellHeight()*sizeof(float), cudaMemcpyHostToDevice);
 
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  //package materials 
  material* cudaMat = NULL;
  cudaMalloc((void**)&cudaMat, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudaMat, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package light
  int* cudaLights = NULL;
  cudaMalloc((void**)&cudaLights, numberOfLights*sizeof(int));
  cudaMemcpy( cudaLights, lightList, numberOfLights*sizeof(int), cudaMemcpyHostToDevice);

  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;


  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudaMat, numberOfMaterials, cudaLights, numberOfLights,cudaRed,cudaGreen,cudaBlue,Input.TellWidth(), Input.TellHeight());

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;
  //free material
  cudaFree( cudaMat );
  cudaFree(cudaLights);
  delete lightList;
  cudaFree(cudaRed);
  delete redInput;
  cudaFree(cudaGreen);
  delete greenInput;
  cudaFree(cudaBlue);
  delete blueInput;
 
 // delete Input;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
