#include<iostream>
#include <fstream>      // std::ifstream
#include <unordered_set>

#include <DGtal/base/Common.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/helpers/ShortcutsGeometry.h>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>


#include "CLI11.hpp"


using namespace DGtal;

// Using standard 3D digital space.
typedef Shortcuts<Z3i::KSpace>         SH3;
typedef ShortcutsGeometry<Z3i::KSpace> SHG3;



void callback()
{
  
}

int main(int argc, char**argv)
{
  CLI::App app{"dgtalpolyscopetemplate"};
  
  
  CLI11_PARSE(app, argc, argv);
  
    polyscope::init();
    polyscope::state::userCallback = callback;
  
 
    polyscope::show();

  return EXIT_SUCCESS;
}
