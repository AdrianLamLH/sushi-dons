"use client";
import { MinusIcon, PlusIcon, Share2Icon, Wand2 } from "lucide-react"
import Image from "next/image"
import { useState, useEffect } from "react"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
} from "@/components/ui/card"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export default function Home() {
  // Initialize state with better structure
  const [isGenerating, setIsGenerating] = useState({ us: false, jp: false })
  const [generationError, setGenerationError] = useState(null)

  const [tagState, setTagState] = useState({
    currentLocation: null,
    tags: {
      us: null,
      jp: null
    },
    descriptions: {
      us: null,
      jp: null
    }
  })
  
  const generateTags = async (location) => {
    setIsGenerating(prev => ({ ...prev, [location]: true }))
    setGenerationError(null)
    
    try {
      const imageUrl = `https://test-sushi-122.s3.us-east-1.amazonaws.com/IMG_5528.jpeg?...`
      
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: imageUrl,
          location: location
        })
      })
      
      if (!response.ok) {
        throw new Error('Failed to generate tags')
      }
      
      const data = await response.json()
      console.log('Received data:', data)
  
      if (data.tags && Object.keys(data.tags).length > 0) {
        const firstProductKey = Object.keys(data.tags)[0]
        const newTags = data.tags[firstProductKey]
        console.log(`Setting ${location} tags:`, newTags)
        
        setTagState(prev => ({
          currentLocation: location,
          tags: {
            ...prev.tags,
            [location]: newTags
          },
          descriptions: {
            ...prev.descriptions,
            [location]: data.description
          }
        }))
      } else {
        throw new Error('Invalid tag data received')
      }
      
    } catch (error) {
      console.error('Error generating tags:', error)
      setGenerationError(`Failed to generate tags: ${error.message}`)
    } finally {
      setIsGenerating(prev => ({ ...prev, [location]: false }))
    }
  }

  const getBadgeVariant = (score) => {
    if (score >= 0.8) {
      return 'high'
    } else if (score >= 0.4) {
      return 'average'
    } else {
      return 'low'
    }
  }

  const renderTags = (tagGroup) => {
    // Only proceed if we have a current location and tags for that location
    console.log(`Received data for ${location}:`, tagState);

    if (!tagState.currentLocation || !tagState.tags[tagState.currentLocation]) {
      return null;
    }

    const currentTags = tagState.tags[tagState.currentLocation];
    
    if (!currentTags[tagGroup]) {
      return null;
    }

    return Object.entries(currentTags[tagGroup]).map(([tag, metrics]) => (
      <TooltipProvider key={tag}>
        <Tooltip>
          <TooltipTrigger>
            <Badge 
              variant={getBadgeVariant(metrics.seo_score)}
              className="cursor-help"
            >
              {tag.replace(/_/g, ' ')}
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <div className="text-sm">
              <div>SEO Score: {(metrics.seo_score * 100).toFixed(0)}%</div>
              <div>Buy Rate: {(metrics.buy_rate * 100).toFixed(0)}%</div>
              <div>Click Rate: {(metrics.click_rate * 100).toFixed(0)}%</div>
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    ))
  }

  return (
    <div className="grid grid-rows-[20px_1fr_20px] min-h-screen font-[family-name:var(--font-geist-sans)]">
      <main className="row-start-2 mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 w-full">
        <div className="grid gap-8 md:grid-cols-2">
          <Card className="overflow-hidden border-0 shadow-none">
            <CardContent className="p-0">
              <div className="aspect-square relative overflow-hidden rounded-lg bg-gray-100">
                <Image
                  src="https://test-sushi-122.s3.us-east-1.amazonaws.com/IMG_5528.jpeg"
                  alt="Short Sleeve Shirt"
                  className="object-cover object-center"
                  fill
                  priority
                />
              </div>
            </CardContent>
          </Card>
          
          <div className="flex flex-col gap-4">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">Short Sleeve Shirt</h1>
              <p className="text-3xl font-bold">$10,000.00 USD</p>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge variant="secondary">Sold Out</Badge>
              {/* <Badge variant="outline">Designer</Badge>
              <Badge variant="outline">Limited Edition</Badge> */}
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <label htmlFor="quantity" className="font-medium">
                  Quantity
                </label>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    disabled
                  >
                    <MinusIcon className="h-4 w-4" />
                    <span className="sr-only">Decrease quantity</span>
                  </Button>
                  <input
                    type="number"
                    id="quantity"
                    className="h-8 w-16 rounded-md border border-input bg-background px-3 py-1 text-center text-sm"
                    min="1"
                    value="1"
                    readOnly
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    disabled
                  >
                    <PlusIcon className="h-4 w-4" />
                    <span className="sr-only">Increase quantity</span>
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Button className="w-full" disabled>
                  Sold out
                </Button>
                <Button className="w-full" variant="outline">
                  Buy with ShopPay
                </Button>
                <Button variant="outline" className="w-full">
                  More payment options
                </Button>
              </div>

              <div className="space-y-4">
              {/* Tag Generation Buttons */}
              <div className="flex justify-end gap-2">
                <Button
                  onClick={() => generateTags('us')}
                  disabled={isGenerating.us || isGenerating.jp}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  <Wand2 className="h-4 w-4" />
                  {isGenerating.us ? 'Generating...' : 'Generate US Tags'}
                </Button>
                <Button
                  onClick={() => generateTags('jp')}
                  disabled={isGenerating.us || isGenerating.jp}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  <Wand2 className="h-4 w-4" />
                  {isGenerating.jp ? 'Generating...' : 'Generate JP Tags'}
                </Button>
              </div>

              {/* Error Message */}
              {generationError && (
                <p className="text-sm text-red-500">{generationError}</p>
              )}

              {/* Show loading state or tags */}
              {isGenerating[tagState.currentLocation] ? (
                <div className="text-sm text-muted-foreground">Generating tags...</div>
              ) : tagState.currentLocation && tagState.tags[tagState.currentLocation] ? (
                <>
                  <div className="text-sm text-muted-foreground mb-2">
                    Showing {tagState.currentLocation.toUpperCase()} market tags
                  </div>
                  <div className="space-y-3">
                    <h3 className="font-medium">Categories</h3>
                    <div className="flex flex-wrap gap-2">
                      {renderTags('category_tags')}
                    </div>
                  </div>
                  <div className="space-y-3">
                    <h3 className="font-medium">Attributes</h3>
                    <div className="flex flex-wrap gap-2">
                      {renderTags('attribute_tags')}
                    </div>
                  </div>
                  <div className="space-y-3">
                    <h3 className="font-medium">Style</h3>
                    <div className="flex flex-wrap gap-2">
                      {renderTags('style_tags')}
                    </div>
                  </div>
                  <div className="space-y-3">
                    <h3 className="font-medium">Usage</h3>
                    <div className="flex flex-wrap gap-2">
                      {renderTags('usage_tags')}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-sm text-muted-foreground">
                  No tags generated yet. Click one of the generate buttons above.
                </div>
              )}
                
                {/* Description section */}
                <div className="space-y-4">
                  {tagState.currentLocation && tagState.descriptions[tagState.currentLocation] ? (
                    <p className="text-muted-foreground">
                      {tagState.descriptions[tagState.currentLocation]}
                    </p>
                  ) : (
                    <p className="text-muted-foreground">
                      Generic product description.
                    </p>
                  )}

                  <Button variant="ghost" size="sm" className="gap-2">
                    <Share2Icon className="h-4 w-4" />
                    Share
                  </Button>
                </div>

                <Button variant="ghost" size="sm" className="gap-2">
                  <Share2Icon className="h-4 w-4" />
                  Share
                </Button>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="#"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Shipping Info
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="#"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Size Guide
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="#"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Returns Policy →
        </a>
      </footer>
    </div>
  )
}