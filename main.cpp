#include<iostream>
#include <fstream>      // std::ifstream
#include <unordered_set>

#include <DGtal/base/Common.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/helpers/ShortcutsGeometry.h>

#include <DGtal/dec/PolygonalCalculus.h>


#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>


#include "CLI11.hpp"


using namespace DGtal;

// Using standard 3D digital space.
typedef Shortcuts<Z3i::KSpace>         SH3;
typedef ShortcutsGeometry<Z3i::KSpace> SHG3;


// Global vars (for polyscope UI)
SHG3::RealVectors ii_normals;
SHG3::PointelRange pointels;
SHG3::SurfelRange surfels;
CountedPtr<SHG3::SurfaceMesh> primalSurface;
CountedPtr<SHG3::LightDigitalSurface> surface;

polyscope::SurfaceMesh *ps;

float epsilon2 = 0.25;
float epsilon1 = 2.0;
float lambda = 0.025;
float alpha = 0.1;

void callback()
{
  ImGui::SliderFloat("alpha", &alpha, 0, 1.0);
  ImGui::SliderFloat("lambda", &lambda, 0, 1.0);
  ImGui::SliderFloat("e_start", &epsilon1, epsilon1, 2.0);
  ImGui::SliderFloat("e_end", &epsilon2, 0, epsilon1);
  
  if (ImGui::Button("Run"))
  {
    auto params = SH3::defaultParameters()  | SHG3::defaultParameters() | SHG3::parametersGeometryEstimation() | SHG3::parametersATApproximation();
   
    trace.beginBlock("AT normal vectors");
    params("at-epsilon-start", epsilon1 );
    params("at-epsilon",  epsilon2 );
    params("at-lambda", lambda);
    params("at-alpha", alpha);

    SH3::Scalars features(pointels.size());
    auto at_normals = SHG3::getATVectorFieldApproximation(features, pointels.begin(), pointels.end(), surface, surfels, ii_normals, params);
    ps->addFaceVectorQuantity("AT normal vectors", at_normals);
    ps->addVertexScalarQuantity("AT (v)", features);
    trace.endBlock();
    
    trace.beginBlock("Gradient");
    PolygonalCalculus<SH3::RealPoint,SH3::RealVector> calculus(*primalSurface);
    std::vector<PolygonalCalculus<SH3::RealPoint,SH3::RealVector>::Vector> gradients;
    std::vector<PolygonalCalculus<SH3::RealPoint,SH3::RealVector>::Vector> cogradients;
    
    auto phiFace = [&](SH3::SurfaceMesh::Face f){
      Eigen::VectorXd ph(4);
      auto vertices = primalSurface->incidentVertices(f);
      size_t cpt=0;
      for(auto v: vertices)
      {
        ph(cpt) =  features[v];
        ++cpt;
      }
      return  ph;
    };
    for(SH3::SurfaceMesh::Face f=0; f < primalSurface->nbFaces(); ++f)
    {
      PolygonalCalculus<SH3::RealPoint,SH3::RealVector>::Vector ph = phiFace(f);
      PolygonalCalculus<SH3::RealPoint,SH3::RealVector>::Vector grad = calculus.gradient(f) * ph;
      PolygonalCalculus<SH3::RealPoint,SH3::RealVector>::Vector cograd = calculus.coGradient(f) * ph;
      gradients.push_back( grad.normalized() );
      cogradients.push_back( cograd.normalized() );
    }
    ps->addFaceVectorQuantity("gradient v",gradients);
    ps->addFaceVectorQuantity("cogradient v",cogradients);
    trace.endBlock();
  }
}

int main(int argc, char**argv)
{
  CLI::App app{"AT-polyscope"};
  
  std::string inputfilename;
  app.add_option("--input,-i", inputfilename,"Input volfile")->required();

  CLI11_PARSE(app, argc, argv);
  
  polyscope::init();
  polyscope::state::userCallback = callback;
  
  auto params = SH3::defaultParameters()  | SHG3::defaultParameters() | SHG3::parametersGeometryEstimation() | SHG3::parametersATApproximation();
  
  auto binary_image = SH3::makeBinaryImage( inputfilename, params );
  auto K            = SH3::getKSpace( binary_image );
  surface      = SH3::makeLightDigitalSurface( binary_image, K, params );
  surfels         = SH3::getSurfelRange( surface, params );
  pointels        = SH3::getPointelRange(surface);
  primalSurface   = SH3::makePrimalSurfaceMesh(surface);

  trace.info() << "#surfels=" << surfels.size() << std::endl;

  trace.beginBlock("Creating polyscope surface");
  std::vector<std::vector<SH3::SurfaceMesh::Vertex>> faces;
  for(auto face= 0 ; face < primalSurface->nbFaces(); ++face)
    faces.push_back(primalSurface->incidentVertices( face ));
  ps = polyscope::registerSurfaceMesh("Vol file", primalSurface->positions(), faces);
  trace.endBlock();
  
  trace.beginBlock("II normal vectors");
  ii_normals = SHG3::getIINormalVectors(binary_image, surfels, params);
  ps->addFaceVectorQuantity("II normal vectors", ii_normals);
  trace.endBlock();
  
  polyscope::show();
  
  return EXIT_SUCCESS;
}
