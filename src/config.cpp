/*
Copyright (c) 2019 Naomasa Matsubayashi (aka. Fadis)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <filesystem>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <unistd.h>
#include <liblnn/config.h>
namespace liblnn { 
  configs_t parse_configs( int argc, const char *argv[] ) {
    namespace po = boost::program_options;
    po::options_description desc( "Options" );
    unsigned int device_index = 0u;
    std::string dump_file;
    std::string train_data;
    std::string train_label;
    std::string eval_data;
    std::string eval_label;
    unsigned int batch_size = 0u;
    unsigned int hidden_width = 0u;
    unsigned int c1_channels = 0u;
    unsigned int c2_channels = 0u;
    desc.add_options()
      ( "help,h", "show this message" )
      ( "list,l", "show all available devices" )
      ( "device,d", po::value< unsigned int >(&device_index)->default_value( 0u ), "use specific device" )
      ( "validation,v", "use VK_LAYER_LUNARG_standard_validation" )
      ( "dump_file,o", po::value< std::string >(&dump_file)->default_value( "nn.dump" ), "dump file" )
      ( "train_data", po::value< std::string >(&train_data)->default_value( "../../mnist/train-images-idx3-ubyte" ), "train data" )
      ( "train_label", po::value< std::string >(&train_label)->default_value( "../../mnist/train-labels-idx1-ubyte" ), "train label" )
      ( "eval_data", po::value< std::string >(&eval_data)->default_value( "../../mnist/t10k-images-idx3-ubyte" ), "eval data" )
      ( "eval_label", po::value< std::string >(&eval_label)->default_value( "../../mnist/t10k-labels-idx1-ubyte" ), "eval label" )
      ( "batch_size,b", po::value< unsigned int >(&batch_size)->default_value( 32u ), "batch size" )
      ( "hidden_width,e", po::value< unsigned int >(&hidden_width)->default_value( 128u ), "hidden width" )
      ( "c1_channels,i", po::value< unsigned int >(&c1_channels)->default_value( 16u ), "c1 channels" )
      ( "c2_channels,j", po::value< unsigned int >(&c2_channels)->default_value( 32u ), "c2 channels" )
      ( "debug,g", "debug mode" );
    po::variables_map vm;
    po::store( po::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );
    if( vm.count( "help" ) ) {
      std::cout << desc << std::endl;
      exit( 0 );
    }
    return configs_t()
      .set_prog_name( std::filesystem::path( argv[ 0 ] ).filename().native() )
      .set_device_index( device_index )
      .set_list( vm.count( "list" ) )
      .set_validation( vm.count( "validation" ) )
      .set_dump_file( dump_file )
      .set_train_data( train_data )
      .set_train_label( train_label )
      .set_eval_data( eval_data )
      .set_eval_label( eval_label )
      .set_batch_size( batch_size )
      .set_hidden_width( hidden_width )
      .set_c1_channels( c1_channels )
      .set_c2_channels( c2_channels )
      .set_debug_mode( vm.count( "debug" ) );
  }
}

