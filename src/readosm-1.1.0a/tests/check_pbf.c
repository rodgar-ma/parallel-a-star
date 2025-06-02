/* 
/ check_pbf.c
/
/ Test cases for OSM .pbf (Protocol Buffer) input file
/
/ Author: Sandro Furieri a.furieri@lqt.it
/
/ ------------------------------------------------------------------------------
/ 
/ Version: MPL 1.1/GPL 2.0/LGPL 2.1
/ 
/ The contents of this file are subject to the Mozilla Public License Version
/ 1.1 (the "License"); you may not use this file except in compliance with
/ the License. You may obtain a copy of the License at
/ http://www.mozilla.org/MPL/
/ 
/ Software distributed under the License is distributed on an "AS IS" basis,
/ WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
/ for the specific language governing rights and limitations under the
/ License.
/
/ The Original Code is the ReadOSM library
/
/ The Initial Developer of the Original Code is Alessandro Furieri
/ 
/ Portions created by the Initial Developer are Copyright (C) 2012-2017
/ the Initial Developer. All Rights Reserved.
/ 
/ Contributor(s):
/
/ Alternatively, the contents of this file may be used under the terms of
/ either the GNU General Public License Version 2 or later (the "GPL"), or
/ the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
/ in which case the provisions of the GPL or the LGPL are applicable instead
/ of those above. If you wish to allow use of your version of this file only
/ under the terms of either the GPL or the LGPL, and not to allow others to
/ use your version of this file under the terms of the MPL, indicate your
/ decision by deleting the provisions above and replace them with the notice
/ and other provisions required by the GPL or the LGPL. If you do not delete
/ the provisions above, a recipient may use your version of this file under
/ the terms of any one of the MPL, the GPL or the LGPL.
/ 
*/

#include <stdio.h>

#include "readosm.h"

static int
check_node (const void *user_data, const readosm_node * node)
{
/* Node callback function: does absolutely nothing */
    if (user_data != NULL || node == NULL)
	user_data = NULL;	/* silencing stupid compiler warnings */
    return READOSM_OK;
}

static int
check_way (const void *user_data, const readosm_way * way)
{
/* Way callback function: does absolutely nothing */
    if (user_data != NULL || way == NULL)
	user_data = NULL;	/* silencing stupid compiler warnings */
    return READOSM_OK;
}

static int
check_relation (const void *user_data, const readosm_relation * relation)
{
/* Relation callback function: does absolutely nothing */
    if (user_data != NULL || relation == NULL)
	user_data = NULL;	/* silencing stupid compiler warnings */
    return READOSM_OK;
}

int
main (int argc, char *argv[])
{
    const void *handle;
    int ret;

    if (argc < 0 || argv == NULL)
	argc = 0;		/* silencing stupid compiler warnings */

    ret = readosm_open ("testdata/test.osm.pbf", &handle);
    if (ret != READOSM_OK)
      {
	  fprintf (stderr, "OPEN ERROR #1: %d\n", ret);
	  return -1;
      }

    ret =
	readosm_parse (handle, (const void *) 0, check_node, check_way,
		       check_relation);
    if (ret != READOSM_OK)
      {
	  fprintf (stderr, ".pbf PARSE error #1: %d\n", ret);
	  return -2;
      }

    ret = readosm_close (handle);
    if (ret != READOSM_OK)
      {
	  fprintf (stderr, "CLOSE ERROR #1: %d\n", ret);
	  return -3;
      }

    ret = readosm_open ("testdata/noNodesPackedInfos.osm.pbf", &handle);
    if (ret != READOSM_OK)
      {
	  fprintf (stderr, "OPEN ERROR #2: %d\n", ret);
	  return -4;
      }

    ret =
	readosm_parse (handle, (const void *) 0, check_node, check_way,
		       check_relation);
    if (ret != READOSM_OK)
      {
	  fprintf (stderr, ".pbf PARSE error #2: %d\n", ret);
	  return -5;
      }

    ret = readosm_close (handle);
    if (ret != READOSM_OK)
      {
	  fprintf (stderr, "CLOSE ERROR #2: %d\n", ret);
	  return -6;
      }

    return 0;
}
