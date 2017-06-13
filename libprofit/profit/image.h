//
// Image class definition
//
// ICRAR - International Centre for Radio Astronomy Research
// (c) UWA - The University of Western Australia, 2017
// Copyright by UWA (in the framework of the ICRAR)
// All rights reserved
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA 02111-1307  USA
//

#ifndef PROFIT_IMAGE_H
#define PROFIT_IMAGE_H

#include <vector>

namespace profit {

template <typename T>
class _2ddata {

public:
	inline
	_2ddata(const std::vector<T> &data, unsigned int width, unsigned int height) :
		data(data.begin(), data.end()),
		width(width),
		height(height),
		size(width * height)
	{
		if (size != data.size()) {
			throw std::invalid_argument("data.size() != this->size");
		}
	}

	std::vector<T> data;
	unsigned int width;
	unsigned int height;
	unsigned int size;

};

typedef _2ddata<bool> Mask;
typedef _2ddata<double> Image;

}  // namespace profit

#endif // PROFIT_IMAGE_H